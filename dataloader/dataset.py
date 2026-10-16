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
from .augmentation import data_augmentation


def collate_fn_dict(batch, is_train=False, noise_factor=0.0002, scaling_factor=0.1):
    collated = default_collate(batch)

    if is_train and 'data' in collated:
        collated['data'] = data_augmentation(collated['data'], noise_factor, scaling_factor)
    return dict(collated)

class IOSDataset(Dataset):
    def __init__(self, data_dict, indices):
        self.indices = indices
        self.data_dict = data_dict


        data_array = data_dict['data_array']
        phase_array = data_dict['phase_array']
        demographic_normalized_array = data_dict['demographic_normalized_array']
        imped_array = data_dict['impedance_array']
        idx_array = data_dict['idx_array']
        if 'sig_feat' in data_dict:
            sig_feat = data_dict['sig_feat']
        else:
            sig_feat = np.zeros_like(idx_array)
        label_array = data_dict['label_array']
        result_array = data_dict['result_array']

        idx_mask = np.isin(idx_array, self.indices)


        self.data_tensor = torch.tensor(data_array[idx_mask], dtype=torch.float32)
        self.phase_tensor = torch.tensor(phase_array[idx_mask], dtype=torch.float32)
        self.demographic_normalized_tensor = torch.tensor(demographic_normalized_array[idx_mask], dtype=torch.float32)
        self.imped_tensor = torch.tensor(imped_array[idx_mask], dtype=torch.float32)
        self.label_tensor = torch.tensor(label_array[idx_mask], dtype=torch.long)
        self.sig_feat_tensor = torch.tensor(sig_feat[idx_mask], dtype=torch.float32)
        self.result_tensor = torch.tensor(result_array[idx_mask], dtype=torch.float32)
        self.idx_tensor = torch.tensor(idx_array[idx_mask], dtype=torch.long)

    def __len__(self):
        return self.data_tensor.shape[0]

    def __patient_num__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = {
            'data': self.data_tensor[idx],
            'phase': self.phase_tensor[idx],
            'demo': self.demographic_normalized_tensor[idx],
            'impedance': self.imped_tensor[idx],
            'label': self.label_tensor[idx],
            'idx': self.idx_tensor[idx],
            'sig_feat': self.sig_feat_tensor[idx],
            'result': self.result_tensor[idx]
        }

        return data
