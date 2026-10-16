import copy, math, time
from typing import *
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import *
from tqdm import tqdm
from scipy.signal import find_peaks


def get_snr(y_true, y_pred):


    signal_power = np.mean(y_true ** 2)

    noise_power = np.mean((y_true - y_pred) ** 2)

    if noise_power < 1e-10:
        return 100.0

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def get_peak_mae(y_true, y_pred):


    peaks, _ = find_peaks(y_true, height=np.mean(y_true), distance=50)

    if len(peaks) == 0:

        return np.mean(np.abs(y_true - y_pred))


    true_peaks = y_true[peaks]
    pred_peaks = y_pred[peaks]


    peak_mae = np.mean(np.abs(true_peaks - pred_peaks))
    return peak_mae

def compute_metrics(y_true, y_pred, mask=None):


    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()


    if mask is not None:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()


        patch_len = y_true.shape[-1] // mask.shape[-1]


        if mask.ndim == 2:
            mask = mask[:, np.newaxis, :]

        full_mask = np.repeat(mask, patch_len, axis=-1)


        full_mask = full_mask[..., :y_true.shape[-1]]


        y_true_eval = y_true[full_mask == 1]
        y_pred_eval = y_pred[full_mask == 1]
    else:

        y_true_eval = y_true
        y_pred_eval = y_pred


    if y_true_eval.size == 0:
        return {'mae': 0, 'mse': 0, 'r2': 0, 'pmae': 0, 'snr': 0}

    mae = np.mean(np.abs(y_true_eval - y_pred_eval))
    mse = np.mean((y_true_eval - y_pred_eval) ** 2)


    try:
        if mask is not None:

            r2 = r2_score(y_true_eval, y_pred_eval)
        else:
            r2 = r2_score(y_true.reshape(-1, y_true.shape[-1]).T,
                          y_pred.reshape(-1, y_pred.shape[-1]).T)
    except:
        r2 = 0.0


    all_pmae = []
    all_snr = []


    if mask is None:
        for b in range(y_true.shape[0]):
            all_pmae.append(get_peak_mae(y_true[b, 0], y_pred[b, 0]))
            all_snr.append(get_snr(y_true[b, 0], y_pred[b, 0]))
    else:

        noise = y_true_eval - y_pred_eval
        signal_power = np.mean(y_true_eval ** 2)
        noise_power = np.mean(noise ** 2)
        snr_val = 10 * np.log10(signal_power / (noise_power + 1e-10))
        all_snr = [snr_val]
        all_pmae = [mae]

    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'pmae': np.mean(all_pmae),
        'snr': np.mean(all_snr)
    }
