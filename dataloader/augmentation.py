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


def data_augmentation(cat_data, noise_factor=0.001, scaling_factor_range=0.1):

    aug_data = cat_data.clone()


    device = aug_data.device
    B, L, C = aug_data.shape


    noise = torch.randn_like(aug_data) * noise_factor
    aug_data += noise


    scales = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * scaling_factor_range
    aug_data *= scales

    return aug_data
