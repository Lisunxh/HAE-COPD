import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import *
from tqdm import tqdm

def compute_metrics_classify(labels, probs, thresh=0.5):
    y = np.asarray(labels).astype(int).reshape(-1)
    p = to_pos_probs(probs)
    pred = (p >= thresh).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    (tn, fp, fn, tp) = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-08)
    sens = tp / (tp + fn + 1e-08)
    spec = tn / (tn + fp + 1e-08)
    ppv = tp / (tp + fp + 1e-08)
    npv = tn / (tn + fn + 1e-08)
    f1 = f1_score(y, pred, zero_division=0)
    gmean = float(np.sqrt(sens * spec))
    try:
        auroc = float(roc_auc_score(y, p))
    except Exception:
        auroc = 0.0
    try:
        (prec_curve, rec_curve, _) = precision_recall_curve(y, p)
        prauc = float(auc(rec_curve, prec_curve))
    except Exception:
        prauc = 0.0
    return {'accuracy': float(acc), 'sens': float(sens), 'spec': float(spec), 'f1': float(f1), 'auc': float(auroc), 'pr_auc': float(prauc), 'g_mean': float(gmean), 'precision': float(ppv), 'ppv': float(ppv), 'npv': float(npv)}

def validate_classify(model, val_loader, device, ema=None, voting=False, return_preds=False, thresh=0.5):
    model.eval()
    if ema is not None:
        ema.store()
        ema.copy_to()
    all_probs = []
    all_labels = []
    all_pids = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['data'].to(device)
            p = batch['phase'].to(device)
            d = batch['demo'].to(device)
            z = batch['impedance'].to(device)
            y = batch['label'][:, 0].to(device).long()
            pid = batch['idx']
            outputs = model(x, phase=p, demo=d, imped=z)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            if torch.is_tensor(pid):
                all_pids.append(pid.detach().cpu().numpy())
            else:
                all_pids.append(np.asarray(pid))
    probs_np = np.vstack(all_probs)
    labels_np = np.concatenate(all_labels).astype(int)
    pid_np = np.concatenate(all_pids)
    if ema is not None:
        ema.restore()
    if voting:
        pos = to_pos_probs(probs_np)
        df = pd.DataFrame({'pid': pid_np, 'y': labels_np, 'p': pos})
        grouped = df.groupby('pid', as_index=False).agg(y=('y', 'max'), p=('p', 'mean'))
        pid_pt = grouped['pid'].values
        y_pt = df.groupby('pid')['y'].max().values.astype(int)
        p_pt = df.groupby('pid')['p'].mean().values.astype(float)
        metrics = compute_metrics_classify(y_pt, p_pt, thresh=thresh)
        if return_preds:
            return (metrics, pid_pt, y_pt, p_pt)
        else:
            return metrics
    else:
        pos = to_pos_probs(probs_np)
        metrics = compute_metrics_classify(labels_np, pos, thresh=thresh)
        if return_preds:
            return (metrics, pid_np, labels_np, pos)
        else:
            return metrics

def to_pos_probs(probs):
    probs = np.asarray(probs)
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1].astype(float)
    return probs.reshape(-1).astype(float)

def find_best_threshold(labels, probs, metric='youden', min_t=0.0, max_t=1.0, n_grid=201):
    y = np.asarray(labels).astype(int).reshape(-1)
    p = to_pos_probs(probs)
    if len(np.unique(y)) < 2:
        return (0.5, float('nan'))
    metric = metric.lower()
    if metric in ['youden', 'j', 'youden_index']:
        (fpr, tpr, thr) = roc_curve(y, p)
        thr = np.asarray(thr)
        valid = np.isfinite(thr)
        (fpr, tpr, thr) = (fpr[valid], tpr[valid], thr[valid])
        in_range = (thr >= min_t) & (thr <= max_t)
        if in_range.sum() > 0:
            (fpr, tpr, thr) = (fpr[in_range], tpr[in_range], thr[in_range])
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        return (float(thr[idx]), float(youden[idx]))
    grid = np.linspace(min_t, max_t, n_grid)
    if metric == 'f1':
        (best_t, best_s) = (0.5, -1)
        for t in grid:
            pred = (p >= t).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            (tn, fp, fn, tp) = cm.ravel()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-08)
            if f1 > best_s:
                (best_s, best_t) = (f1, t)
        return (float(best_t), float(best_s))
    if metric in ['gmean', 'g-mean', 'g_mean']:
        (best_t, best_s) = (0.5, -1)
        for t in grid:
            pred = (p >= t).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            (tn, fp, fn, tp) = cm.ravel()
            sens = tp / (tp + fn + 1e-08)
            spec = tn / (tn + fp + 1e-08)
            g = np.sqrt(sens * spec)
            if g > best_s:
                (best_s, best_t) = (g, t)
        return (float(best_t), float(best_s))
    raise ValueError("metric must be one of: 'youden', 'f1', 'gmean'")

class EarlyStopping:

    def __init__(self, patience=7, mode='max', delta=0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max' and score < self.best_score + self.delta or (self.mode == 'min' and score > self.best_score - self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_and_validate_classify(model, train_loader, val_loader, epochs, optimizer, scheduler, criterion, device='cpu', valid_interval=1, save_model=False, model_path=None, ema=None, voting=False, patience=7):
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, mode='max')
    history = {'train_loss': [], 'val_f1': [], 'val_pr_auc': [], 'val_auc': [], 'val_sens': [], 'val_spec': [], 'val_g_mean': []}
    best_val_metric = -np.inf
    best_model_state = None
    print(f'🚀 Starting Training | Device: {device}')
    try:
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                x = batch['data'].to(device)
                p = batch['phase'].to(device)
                d = batch['demo'].to(device)
                z = batch['impedance'].to(device)
                targets = batch['label'][:, 0].to(device, non_blocking=True).long()
                optimizer.zero_grad()
                outputs = model(x, phase=p, demo=d, imped=z)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                if ema is not None:
                    ema.update()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            if scheduler is not None and (not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                scheduler.step()
            if (epoch + 1) % valid_interval == 0:
                metrics = validate_classify(model, val_loader, device, ema=ema, voting=voting)
                history['val_f1'].append(metrics['f1'])
                history['val_pr_auc'].append(metrics['pr_auc'])
                history['val_auc'].append(metrics['auc'])
                history['val_sens'].append(metrics['sens'])
                history['val_spec'].append(metrics['spec'])
                history['val_g_mean'].append(metrics['g_mean'])
                print(f'Epoch: {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f}')
                print(f"Metrics: AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f}")
                print(f"Detailed: Sens: {metrics['sens']:.4f} | Spec: {metrics['spec']:.4f} | G-Mean: {metrics['g_mean']:.4f}")
                print('-' * 60)
                if metrics['pr_auc'] > best_val_metric:
                    best_val_metric = metrics['pr_auc']
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f'⭐ New Best Model Found (pr_AUC: {best_val_metric:.4f})')
                early_stopping(metrics['pr_auc'])
                if early_stopping.early_stop:
                    print(f'🛑 Early stopping triggered at epoch {epoch + 1}')
                    break
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    if save_model and best_model_state:
        torch.save(best_model_state, model_path)
        print(f'💾 Best model saved to {model_path}')
    return (history, best_model_state)

def get_predictions_classify(model, dataloader, device, voting=True):
    model.eval()
    model.to(device)
    all_probs = []
    all_targets = []
    all_idxs = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Extracting predictions')
        for batch in pbar:
            x = batch['data'].to(device)
            p = batch['phase'].to(device)
            d = batch['demo'].to(device)
            z = batch['impedance'].to(device)
            targets = batch['label'][:, 0].to(device).long()
            idxs = batch['idx']
            outputs = model(x, p, d, z)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_idxs.extend(idxs.cpu().numpy().tolist())
    all_probs_np = np.vstack(all_probs)
    all_targets_np = np.array(all_targets)
    all_idxs_np = np.array(all_idxs)
    if not voting:
        return (all_targets_np, all_probs_np)
    df = pd.DataFrame({'patient_id': all_idxs_np, 'label': all_targets_np})
    prob_cols = [f'prob_class_{i}' for i in range(all_probs_np.shape[1])]
    df[prob_cols] = all_probs_np
    patient_results = df.groupby('patient_id').agg({**{col: 'mean' for col in prob_cols}, 'label': 'first'}).reset_index()
    final_patient_probs = patient_results[prob_cols].values
    final_patient_labels = patient_results['label'].values
    return (final_patient_labels, final_patient_probs)

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
