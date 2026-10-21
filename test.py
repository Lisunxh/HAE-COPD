import sys
sys.dont_write_bytecode = True

from train import ROOT, load_data, loader, autoencoder, classifier, file_hash, output_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.model_selection import StratifiedKFold
from training_utils import get_predictions_classify, compute_metrics_classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--internal')
    parser.add_argument('--external')
    parser.add_argument('--output', default='outputs/test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if not args.internal and not args.external:
        parser.error('Provide --internal and/or --external')
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    checkpoint_dir = Path(args.checkpoints).resolve()
    config = json.loads((checkpoint_dir / 'config.json').read_text(encoding='utf-8'))
    data = {}
    if args.internal:
        if file_hash(args.internal) != config['internal_sha256']:
            raise ValueError('Internal input differs from training input')
        data['internal'] = load_data(args.internal)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=config['seed']).split(
            np.zeros(len(data['internal']['stratify'])), data['internal']['stratify']))
    if args.external:
        data['external'] = load_data(args.external)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    out = output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for fold in config['folds']:
        checkpoint = checkpoint_dir / f'classify_f{fold}.pth'
        threshold_info = json.loads((checkpoint_dir / f'threshold_f{fold}.json').read_text(encoding='utf-8'))
        if threshold_info['checkpoint_sha256'] != file_hash(checkpoint):
            raise ValueError('Threshold/checkpoint mismatch')
        threshold = float(threshold_info['youden'])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid saved threshold')
        torch.manual_seed(66)
        model = classifier(autoencoder()).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        for cohort, dataset in data.items():
            indices = folds[fold - 1][1] if cohort == 'internal' else np.arange(len(dataset['stratify']))
            val_loader = loader(dataset, indices, args.batch_size)
            y, probability = get_predictions_classify(model, val_loader, device, voting=True)
            if len(np.unique(y)) != 2:
                raise ValueError('Evaluation requires both classes')
            for name, threshold_value in [('youden', threshold), ('fixed', 0.5)]:
                metrics = compute_metrics_classify(y, probability, threshold_value)
                results.append({'cohort': cohort, 'fold': fold, 'records': len(y),
                                'threshold_type': name, 'threshold': threshold_value, **metrics})
                print(cohort, fold, name, metrics)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    frame = pd.DataFrame(results)
    frame.to_csv(out / 'fold_metrics.csv', index=False)
    summary = []
    metrics = ['accuracy', 'auc', 'pr_auc', 'sens', 'spec', 'f1', 'ppv', 'npv', 'g_mean']
    for (cohort, threshold_type), group in frame.groupby(['cohort', 'threshold_type']):
        for metric in metrics:
            values = group[metric].to_numpy()
            mean = float(values.mean())
            half = float(student_t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5)) if len(values) == 5 else np.nan
            summary.append({'cohort': cohort, 'threshold_type': threshold_type,
                            'metric': metric, 'folds': len(values), 'mean': mean,
                            'ci_low': mean - half, 'ci_high': mean + half})
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    (out / 'evaluation.json').write_text(json.dumps({
        'device': device, 'batch_size': args.batch_size,
        'torch': str(torch.__version__),
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'aggregation': 'recording_mean_probability',
        'ci': 'five-fold mean +/- t(4) * sample SD / sqrt(5); omitted for partial folds',
        'external_threshold_selection': False}, indent=2), encoding='utf-8')
    print('Saved:', out)


if __name__ == '__main__':
    main()
