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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BioPhasePositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, patch_emb, patch_radians):

        B, N = patch_radians.shape
        device = patch_emb.device

        sin_part = torch.sin(patch_radians).unsqueeze(-1)
        cos_part = torch.cos(patch_radians).unsqueeze(-1)

        phase_pe = torch.zeros_like(patch_emb, device=device)
        phase_pe[:, :, 0::2] = sin_part
        phase_pe[:, :, 1::2] = cos_part

        return patch_emb + self.alpha * phase_pe

class MultiFrequencyBioEmbedding(nn.Module):
    def __init__(self, d_model, min_freq=0.001, max_freq=2.0, dropout=0.1):

        super().__init__()
        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        num_freqs = d_model // 2
        exponent = torch.linspace(0, 1, num_freqs)

        freqs = min_freq * torch.pow((max_freq / min_freq), exponent)

        self.register_buffer('freq_weight', freqs)


    def forward(self, patch_emb, patch_radians):

        device = patch_emb.device

        scaled_radians = patch_radians.unsqueeze(-1) * self.freq_weight

        phase_pe = torch.zeros_like(patch_emb, device=device)
        phase_pe[:, :, 0::2] = torch.sin(scaled_radians)
        phase_pe[:, :, 1::2] = torch.cos(scaled_radians)

        patch_emb = patch_emb + phase_pe

        return self.dropout(patch_emb)

class MixedPositionalPhaseEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000,
                 min_freq=0.001, max_freq=2.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_pe", pe)

        num_freqs = d_model // 2
        exponent = torch.linspace(0, 1, num_freqs)
        freqs = min_freq * torch.pow((max_freq / min_freq), exponent)
        self.register_buffer("freq_weight", freqs)

        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, patch_emb: torch.Tensor, patch_radians: torch.Tensor = None) -> torch.Tensor:
        B, N, D = patch_emb.shape
        device = patch_emb.device


        pos_pe = self.pos_pe[:, :N, :]

        emb = patch_emb + self.alpha * pos_pe


        scaled_radians = patch_radians.unsqueeze(-1) * self.freq_weight
        phase_pe = torch.zeros_like(patch_emb)
        phase_pe[:, :, 0::2] = torch.sin(scaled_radians)
        phase_pe[:, :, 1::2] = torch.cos(scaled_radians)
        emb = emb + self.beta * phase_pe

        return self.dropout(emb)
