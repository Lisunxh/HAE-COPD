import sys
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from runtime import configure
ROOT = configure()

import argparse
import hashlib
import json
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from sklearn.model_selection import StratifiedKFold
from dataloader.dataset import IOSDataset, collate_fn_dict
from models.hae import Hierarchical_IOSAE
from models.classifier import IOSClassifier, OptimizedResidualGatedHead
from training_utils import FocalLoss, train_and_validate_classify, get_predictions_classify, find_best_threshold, compute_metrics_classify


def load_data(path):
    data = np.load(path, allow_pickle=True).item()
    n = len(data['idx_array'])
    shapes = {'data_array': (n, 2, 4096), 'phase_array': (n, 4096),
              'demographic_normalized_array': (n, 5), 'impedance_array': (n, 8)}
    for key, shape in shapes.items():
        if np.asarray(data[key]).shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key}; expected {shape}')
    labels = np.asarray(data['label_array'])
    if labels.ndim != 2 or len(labels) != n or not np.isin(labels[:, 0], [0, 1]).all():
        raise ValueError('label_array must contain binary labels in column 0')
    ids = np.asarray(data['idx_array'])
    if not np.array_equal(np.unique(ids), np.arange(len(data['stratify']))):
        raise ValueError('idx_array must index stratify records consecutively from zero')
    if pd.DataFrame({'id': ids, 'label': labels[:, 0]}).groupby('id').label.nunique().max() != 1:
        raise ValueError('Inconsistent labels within a record')
    return data


def loader(data, indices, batch_size, shuffle=False, augment=False):
    return DataLoader(IOSDataset(data, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      collate_fn=partial(collate_fn_dict, is_train=augment))


def autoencoder(mask_mode='none'):
    return Hierarchical_IOSAE(in_channels=2, input_length=4096, patch_len=16,
                             d_model=256, mask_mode=mask_mode, mask_ratio=0.75,
                             embed_mode='bioembed', mask_span=8)


def classifier(base):
    return IOSClassifier(base.encoder, OptimizedResidualGatedHead(
        d_model=512, d_demo=5, d_imped=8, d_plus=0, num_classes=2,
        dropout=0.2), freeze_encoder=False)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Outputs must stay inside this package')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--internal', required=True)
    parser.add_argument('--output', default='outputs/train')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--pretrain-epochs', type=int, default=90)
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.finetune_epochs, args.batch_size) < 1:
        parser.error('Epochs and batch size must be positive')
    if len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(1, 6)):
        parser.error('Choose unique fold numbers from 1 to 5')
    data = load_data(args.internal)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=66).split(
        np.zeros(len(data['stratify'])), data['stratify']))
    for train_idx, val_idx in folds:
        for indices in (train_idx, val_idx):
            labels = data['label_array'][np.isin(data['idx_array'], indices), 0]
            if len(np.unique(labels)) != 2:
                raise ValueError('Each training and validation fold needs both classes')
    out = output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = {k: v for k, v in vars(args).items() if k not in ('internal', 'output')}
    config.update(seed=66, internal_sha256=file_hash(args.internal),
                  python=sys.version.split()[0], torch=str(torch.__version__),
                  cudnn_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  aggregation='recording_mean_probability', device=device)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    torch.manual_seed(66)
    np.random.seed(66)
    results = []
    for fold in args.folds:
        train_idx, val_idx = folds[fold - 1]
        train_loader = loader(data, train_idx, args.batch_size, shuffle=True)
        val_loader = loader(data, val_idx, args.batch_size)
        model_ae = autoencoder('response').to(device)
        optimizer = torch.optim.Adam(model_ae.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        history, state = model_ae.train_and_validate(
            model_ae, train_loader, val_loader, args.pretrain_epochs,
            optimizer, scheduler, model_ae.loss_function, device=device)
        if state is None:
            raise RuntimeError('Pretraining produced no checkpoint')
        torch.save(state, out / f'pretrain_f{fold}.pth')
        pd.DataFrame(history).to_csv(out / f'pretrain_history_f{fold}.csv', index=False)
        model_ae.load_state_dict(state)
        del state, optimizer, scheduler
        model_ae.encoder.mask_mode = 'none'
        model = classifier(model_ae).to(device)
        del model_ae
        train_loader = loader(data, train_idx, args.batch_size, shuffle=True, augment=True)
        optimizer = torch.optim.AdamW([
            {'params': model.head.parameters(), 'lr': 8e-5, 'weight_decay': 0.2},
            {'params': model.encoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.08}])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[8e-5, 1e-5], epochs=args.finetune_epochs,
            steps_per_epoch=len(train_loader), pct_start=0.15,
            div_factor=10, final_div_factor=1000, anneal_strategy='cos')
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        history, state = train_and_validate_classify(
            model, train_loader, val_loader, args.finetune_epochs, optimizer,
            scheduler, FocalLoss(alpha=0.7, gamma=2.0), device=device,
            voting=True, ema=ema, patience=8)
        if state is None:
            raise RuntimeError('Fine-tuning produced no checkpoint')
        torch.save(state, out / f'classify_f{fold}.pth')
        pd.DataFrame(history).to_csv(out / f'finetune_history_f{fold}.csv', index=False)
        model.load_state_dict(state)
        y, probability = get_predictions_classify(model, val_loader, device, voting=True)
        threshold, _ = find_best_threshold(y, probability, metric='youden')
        (out / f'threshold_f{fold}.json').write_text(json.dumps(
            {'fold': fold, 'youden': threshold, 'selected_on': 'internal_validation',
             'checkpoint_sha256': file_hash(out / f'classify_f{fold}.pth')}, indent=2), encoding='utf-8')
        for name, t in [('youden', threshold), ('fixed', 0.5)]:
            results.append({'fold': fold, 'threshold_type': name, 'threshold': t,
                            **compute_metrics_classify(y, probability, t)})
        pd.DataFrame(results).to_csv(out / 'internal_metrics.csv', index=False)
        del model, state, optimizer, scheduler, ema, train_loader, val_loader
        if device == 'cuda':
            torch.cuda.empty_cache()
    print('Saved:', out)


if __name__ == '__main__':
    main()
