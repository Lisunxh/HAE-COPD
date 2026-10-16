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


class IOSClassifier(nn.Module):
    def __init__(self, encoder, head, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.head = head

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, phase=None, demo=None, imped=None, Z_plus=None, return_gate=False):
        Z, _ = self.encoder(x, demo, phase)

        logits = self.head(Z, demo=demo, imped=imped, Z_plus=Z_plus, return_gate=return_gate)

        return logits

class OptimizedResidualGatedHead(nn.Module):
    def __init__(self, d_model, d_demo=5, d_imped=8, d_plus=96, num_classes=2, dropout=0.3):
        super().__init__()
        self.d_demo = d_demo
        self.d_imped = d_imped
        self.d_plus = d_plus

        self.signal_attn = nn.Linear(d_model, 1)

        if d_demo:
            self.demo_embed = nn.Sequential(
                nn.Linear(d_demo, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
            self.gate_demo = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )

        if d_imped:
            self.imped_embed = nn.Sequential(
                nn.Linear(d_imped, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
            self.gate_imped = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
        if d_plus:
            self.plus_embed = nn.Sequential(
                nn.Linear(d_plus, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
            self.gate_plus = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.cls = nn.Linear(d_model, num_classes)
        self.modality_dropout = 0.2

    def forward(self, Z, demo=None, imped=None, Z_plus=None, return_gate=False):
        B, N, D = Z.shape

        w = torch.softmax(self.signal_attn(Z), dim=1)
        z_signal = (w * Z).sum(dim=1)
        z_final = z_signal

        g_d = None
        g_i = None
        z_d = None
        z_i = None

        if self.d_demo is not None and demo is not None:
            z_d = self.demo_embed(demo)
            g_d = self.gate_demo(torch.cat([z_signal, z_d], dim=-1))
            z_final = z_final + g_d * z_d

        if self.d_imped is not None and imped is not None:
            z_i = self.imped_embed(imped)
            g_i = self.gate_imped(torch.cat([z_signal, z_i], dim=-1))
            if self.training and torch.rand(1) < self.modality_dropout:
                g_i = g_i * 0
            z_final = z_final + g_i * z_i

        z_final = self.ffn_norm(z_final)
        z_final = z_final + self.ffn(z_final)

        logits = self.cls(z_final)

        if return_gate:
            return {
                "logits": logits,
                "z_signal": z_signal,
                "z_demo": z_d,
                "z_imped": z_i,
                "gate_demo": g_d,
                "gate_imped": g_i
            }

        return logits
