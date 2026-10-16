import sys
sys.dont_write_bytecode = True

from runtime import configure
configure()

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader.dataset import IOSDataset, collate_fn_dict
from models.hae import Hierarchical_IOSAE
from models.classifier import IOSClassifier, OptimizedResidualGatedHead


if __name__ == '__main__':
    torch.manual_seed(66)
    np.random.seed(66)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'data_array': np.random.randn(1, 2, 4096).astype(np.float32),
        'phase_array': np.linspace(0, 4 * np.pi, 4096, dtype=np.float32)[None, :],
        'demographic_normalized_array': np.random.randn(1, 5).astype(np.float32),
        'impedance_array': np.random.randn(1, 8).astype(np.float32),
        'label_array': np.zeros((1, 2), dtype=np.int64),
        'result_array': np.zeros((1, 3), dtype=np.float32),
        'idx_array': np.zeros(1, dtype=np.int64)
    }

    test_subset = IOSDataset(data_dict, np.arange(1))
    test_loader = DataLoader(
        test_subset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda batch: collate_fn_dict(batch, is_train=False)
    )

    model_ae = Hierarchical_IOSAE(
        in_channels=2, input_length=4096, patch_len=16, d_model=256,
        mask_mode='none', mask_ratio=0.75, embed_mode='bioembed', mask_span=8
    )
    model = IOSClassifier(
        model_ae.encoder,
        OptimizedResidualGatedHead(d_model=512, d_demo=5, d_imped=8,
                                   d_plus=0, num_classes=2, dropout=0.2),
        freeze_encoder=False
    ).to(device)
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            data = batch['data'].to(device)
            phase = batch['phase'].to(device)
            demo = batch['demo'].to(device)
            impedance = batch['impedance'].to(device)
            output = model(data, phase=phase, demo=demo, imped=impedance)
            probability = torch.softmax(output, dim=1)
            assert output.shape == (1, 2) and torch.isfinite(output).all()
            assert torch.allclose(probability.sum(dim=1), torch.ones(1, device=device))
            print('Device:', device)
            print('Input:', tuple(data.shape))
            print('Output:', tuple(output.shape))
            print('Probability:', probability.cpu().numpy())
