
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
from .embeddings import *
from .reconstruction import compute_metrics


class ConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_len, d_model):
        super().__init__()

        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=500//(4000//patch_len),
            stride=500//(4000//patch_len),
            bias=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)

class FiLM(nn.Module):
    def __init__(self, meta_dim, feature_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * 2)
        )

    def forward(self, x, meta):


        mod = self.mlp(meta).unsqueeze(-1)
        gamma, beta = torch.chunk(mod, 2, dim=1)
        return (1 + gamma) * x + beta

class ResidualBlockWithFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, meta_dim, stride=2, dropout=0.1, use_film=True):
        super().__init__()
        self.use_film = use_film


        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )


        if self.use_film:
            self.film = FiLM(meta_dim, out_ch)

        self.dropout = nn.Dropout(dropout)


        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, meta):

        out = self.conv(x)


        if self.use_film:
            out = self.film(out, meta)


        return out + self.dropout(self.shortcut(x))

class ConvStem(nn.Module):
    def __init__(self, in_channels, meta_dim, kernel_sizes=[3, 3, 3], channels=[64, 128, 256], dropout=0.1, use_film=True):
        super().__init__()
        self.layers = nn.ModuleList()

        curr_ch = in_channels
        for i in range(len(kernel_sizes)):
            self.layers.append(
                ResidualBlockWithFiLM(
                    curr_ch,
                    channels[i],
                    kernel_sizes[i],
                    meta_dim,
                    stride=2,
                    dropout=dropout,
                    use_film=use_film
                )
            )
            curr_ch = channels[i]

    def forward(self, x, meta):


        for layer in self.layers:
            x = layer(x, meta)
        return x

class SignalMasker(nn.Module):
    def __init__(self, mask_ratio=0.4, num_patches=50, patch_len=80, in_channels=2, mask_span=1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.N = num_patches // mask_span
        self.P = patch_len * mask_span
        self.C = in_channels

    def forward(self, x, mode='consistent'):


        final_mask = None
        if mode == 'none':
            return x, final_mask

        B, C, L = x.shape
        device = x.device
        x = x.reshape(B, C, self.N, self.P)

        if mode == 'consistent':


            mask = self._generate_mask(B, self.N, device)
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
            x = x * (~mask_expanded)

            final_mask = mask.unsqueeze(1).repeat(1, C, 1)

        elif mode == 'independent':


            mask = self._generate_mask(B * C, self.N, device).reshape(B, C, self.N)
            x = x * (~mask.unsqueeze(-1))
            final_mask = mask

        elif mode == 'response':


            mask_resp = self._generate_mask(B, self.N, device)


            final_mask = torch.zeros((B, C, self.N), device=device, dtype=torch.bool)
            final_mask[:, 0, :] = mask_resp

            x = x * (~final_mask.unsqueeze(-1))

        return x.reshape(B, C, -1), final_mask

    def _generate_mask(self, batch_size, num_patches, device):

        len_keep = int(num_patches * (1 - self.mask_ratio))

        if len_keep < 1:
            len_keep = 1
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.ones([batch_size, num_patches], device=device, dtype=torch.bool)
        mask[:, :len_keep] = False


        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

class PatchTemporalConv(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False
        )

    def forward(self, x):

        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        return residual + x.transpose(1, 2)

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class TemporalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.conv_norm = nn.BatchNorm1d(d_model)
        self.temporal_conv = PatchTemporalConv(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),


            Permute(0, 2, 1),
            nn.Conv1d(dim_feedforward, dim_feedforward, kernel_size=3, padding=1, groups=dim_feedforward),
            nn.GELU(),
            Permute(0, 2, 1),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])


        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x

class LocalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, window_size=16):
        super().__init__()
        self.window_size = window_size

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )


        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            Permute(0, 2, 1),
            nn.Conv1d(dim_feedforward, dim_feedforward, kernel_size=3, padding=1, groups=dim_feedforward),
            nn.GELU(),
            Permute(0, 2, 1),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, N, D = x.shape
        w = self.window_size


        num_windows = N // w
        x_norm = self.norm1(x)


        x_windows = x_norm.view(-1, w, D)


        attn_out, _ = self.self_attn(x_windows, x_windows, x_windows)


        x = x + self.dropout(attn_out.reshape(B, N, D))


        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x

class ConvPatchFusion(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.shortcut = nn.Sequential(
            nn.Conv1d(dim, out_dim, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):


        B, N, D = x.shape
        if N % 2 != 0:


            x = F.pad(x, (0, 0, 0, 1))


        x = x.transpose(1, 2)


        identity = self.shortcut(x)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)


        out += identity
        out = self.act(out)


        return out.transpose(1, 2)

class Hierarchical_Transformer_encoder(nn.Module):
    def __init__(self, in_channels, sequence_length, patch_len, d_model, num_layers=4, dropout=0.1, mask_mode='none', mask_ratio=0.4, embed_mode=None, mask_span=1, use_film=True):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.num_patches = sequence_length // patch_len
        self.num_layers = num_layers
        self.sequence_start = sequence_length - self.num_patches * patch_len
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.embed_mode = embed_mode


        self.signal_masker = SignalMasker(
            mask_ratio=mask_ratio,
            num_patches=self.num_patches,
            patch_len=patch_len,
            in_channels=in_channels,
            mask_span=mask_span
        )


        self.stem = ConvStem(in_channels, meta_dim=5, use_film=use_film)
        self.patch_embed = ConvPatchEmbedding(256, patch_len=patch_len, d_model=d_model)

        if self.embed_mode == 'bioembed':
            self.pos_embed = MixedPositionalPhaseEncoding(d_model=d_model, max_len=self.num_patches)
        elif self.embed_mode == 'sinusoidal':
            self.pos_embed = PositionalEncoding(d_model=d_model, max_len=self.num_patches)
        else:
            self.pos_embed = nn.Identity()


        self.stage1_layers = nn.ModuleList([
            TemporalTransformerLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_layers // 2)
        ])


        self.fusion = ConvPatchFusion(d_model, d_model * 2)


        self.stage2_layers = nn.ModuleList([
            TemporalTransformerLayer(d_model=d_model * 2, nhead=8, dim_feedforward=d_model * 8, dropout=dropout)
            for _ in range(num_layers // 2)
        ])

        self.norm = nn.LayerNorm(d_model*2)

    def forward(self, input: Tensor, meta, phase: Tensor = None) -> tuple[Tensor, Tensor]:

        B, C, L = input.shape
        N = self.num_patches
        P = self.patch_len


        x = input[:, :, self.sequence_start:]
        if phase is not None:
            phase = phase[:, self.sequence_start:]


        x, mask = self.signal_masker(x, mode=self.mask_mode)


        x = self.stem(x, meta)
        x = self.patch_embed(x)


        if self.embed_mode == 'bioembed' and phase is not None:
            phase = phase.reshape(B, N, P)
            phase = torch.mean(phase, dim=-1)


        if self.embed_mode == 'bioembed' and phase is not None:
            x = self.pos_embed(x, phase)
        else:
            x = self.pos_embed(x)


        for layer in self.stage1_layers:
            x = layer(x)


        x = self.fusion(x)


        for layer in self.stage2_layers:
            x = layer(x)

        result = self.norm(x)

        return result, mask

class HierarchicalDecoder(nn.Module):
    def __init__(self, d_model, in_channels, patch_len, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.patch_len = patch_len

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model * 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
           )


        self.refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
            )


        self.output_proj = nn.Linear(d_model, in_channels * patch_len)


    def forward(self, x):


        B, _, _ = x.shape
        N, C, P = self.num_patches, self.in_channels, self.patch_len


        x = x.transpose(1, 2)

        x = self.upsample(x)

        if x.shape[-1] > N:
            x = x[:, :, :N]
        elif x.shape[-1] < N:
            x = F.pad(x, (0, N - x.shape[-1]))

        x = self.refine(x)


        x = x.transpose(1, 2)


        x = self.output_proj(x)


        x = x.view(B, N, C, P).transpose(1, 2).reshape(B, C, -1)

        return x

class LinearDecoder(nn.Module):
    def __init__(self, d_model, input_length, in_channels):
        super().__init__()
        self.input_length = input_length
        self.in_channels = in_channels


        self.stride_total = 32


        self.len_feat = input_length//8


        self.expansion_dim = in_channels * self.stride_total


        self.fc = nn.Linear(d_model, self.expansion_dim)

    def forward(self, x):
        print(x.shape)

        B, C_feat, L_feat = x.shape


        x = x.transpose(1, 2)
        x = self.fc(x)


        x = x.view(B, L_feat, self.in_channels, self.stride_total)


        x = x.transpose(1, 2).reshape(B, self.in_channels, -1)


        if x.shape[-1] > self.input_length:
            x = x[:, :, :self.input_length]

        return x

class Hierarchical_IOSAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 input_length: int,
                 patch_len: int,
                 d_model: int,
                 dropout: float = 0.1,
                 mask_mode: str = 'none',
                 mask_ratio: float = 0.4,
                 embed_mode=None,
                 mask_span=1,
                 use_film=True,
                 **kwargs) -> None:
        super(Hierarchical_IOSAE, self).__init__()
        self.num_patches = input_length // patch_len
        self.patch_len = patch_len
        self.in_channels = in_channels
        self.embed_mode = embed_mode
        self.mask_mode = mask_mode

        self.encoder = Hierarchical_Transformer_encoder(in_channels, input_length, patch_len, d_model, num_layers=4, dropout=dropout, mask_mode=mask_mode, mask_ratio=mask_ratio, embed_mode=embed_mode, mask_span=mask_span, use_film=use_film)

        self.decoder = HierarchicalDecoder(d_model, in_channels, patch_len, self.num_patches)

    def forward(self, x, meta, phase=None):


        B, C, L = x.shape


        if self.embed_mode == 'bioembed' and phase is not None:
            z, mask = self.encoder(x, meta, phase)
        else:
            z, mask = self.encoder(x, meta)


        result = self.decoder(z)

        return result, mask

    def loss_function(self, recons, target, mask=None):


        if self.mask_mode != 'none':


            patch_len = target.shape[-1] // mask.shape[-1]
            full_mask = mask.repeat_interleave(patch_len, dim=-1)


            mse_per_point = F.mse_loss(recons, target, reduction='none')


            l2 = (mse_per_point * full_mask).sum() / (full_mask.sum() + 1e-6)
            l1 = (torch.abs(recons - target) * full_mask).sum() / (full_mask.sum() + 1e-6)

            masked_loss = 0.7 * l2 + 0.3 * l1


            global_loss = F.mse_loss(recons, target)


            recon_loss = masked_loss
        else:

            recon_loss = F.mse_loss(recons, target)

        total_loss = recon_loss

        return {
            'loss': total_loss,
            'Reconstruction_Loss': recon_loss
        }

    def train_and_validate(self, model, train_loader, val_loader, epochs, optimizer, scheduler, criterion, device='cpu',
                                 valid_interval=1,
                                 save_model=False, model_path=None):

        model = model.to(device)
        print(f"Using device: {device}")

        best_val_mae = float('inf')
        best_model_state = None

        history = {
            'train_loss': [], 'train_mae': [],
            'val_loss': [], 'val_mae': [], 'val_mse': [],
            'val_r2': [], 'val_pmae': [], 'val_snr': []
        }

        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                model.train()
                total_train_loss = 0
                total_train_mae = 0

                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))
                for batch in pbar:
                    input_tensor = batch['data'].to(device, non_blocking=True)
                    meta_tensor = batch['demo'].to(device, non_blocking=True)

                    optimizer.zero_grad()

                    if self.embed_mode == 'bioembed':
                        phase_tensor = batch['phase'].to(device, non_blocking=True)
                        output_tensor, mask = model(input_tensor, meta_tensor, phase_tensor)
                    else:
                        output_tensor, mask = model(input_tensor, meta_tensor)

                    loss_dict = criterion(output_tensor, input_tensor, mask=mask)
                    loss = loss_dict['loss']
                    loss.backward()
                    optimizer.step()

                    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                    total_train_loss += loss.item()

                    if self.mask_mode != 'none':

                        patch_len = input_tensor.shape[-1] // mask.shape[-1]

                        full_mask = torch.repeat_interleave(mask, patch_len, dim=-1)
                        total_train_mae += torch.mean(torch.abs(output_tensor - input_tensor)[full_mask.bool()]).item()
                    else:
                        total_train_mae += torch.mean(torch.abs(output_tensor - input_tensor)).item()
                    pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

                if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()


                avg_train_loss = total_train_loss / len(train_loader)
                avg_train_mae = total_train_mae / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                history['train_mae'].append(avg_train_mae)


                if (epoch + 1) % valid_interval == 0 or epoch == epochs - 1 or epoch == 0:
                    model.eval()
                    val_batch_metrics = []
                    total_val_loss = 0

                    with torch.no_grad():
                        for batch in val_loader:
                            input_tensor = batch['data'].to(device)
                            meta_tensor = batch['demo'].to(device, non_blocking=True)

                            if self.embed_mode == 'bioembed':
                                phase_tensor = batch['phase'].to(device, non_blocking=True)
                                output_tensor, mask = model(input_tensor, meta_tensor, phase_tensor)
                            else:
                                output_tensor, mask = model(input_tensor, meta_tensor)

                            loss_dict = criterion(output_tensor, input_tensor, mask=mask)
                            total_val_loss += loss_dict['loss'].item()


                            metrics = compute_metrics(input_tensor, output_tensor, mask)
                            val_batch_metrics.append(metrics)


                    avg_val_loss = total_val_loss / len(val_loader)
                    epoch_val_mae = np.mean([m['mae'] for m in val_batch_metrics])
                    epoch_val_snr = np.mean([m['snr'] for m in val_batch_metrics])
                    epoch_val_pmae = np.mean([m['pmae'] for m in val_batch_metrics])
                    epoch_val_r2 = np.mean([m['r2'] for m in val_batch_metrics])


                    history['val_loss'].append(avg_val_loss)
                    history['val_mae'].append(epoch_val_mae)
                    history['val_snr'].append(epoch_val_snr)
                    history['val_pmae'].append(epoch_val_pmae)
                    history['val_r2'].append(epoch_val_r2)
                    history['val_mse'].append(np.mean([m['mse'] for m in val_batch_metrics]))

                    epoch_duration = time.time() - epoch_start
                    print(f"\nEpoch [{epoch + 1}/{epochs}] ({epoch_duration:.1f}s) "
                          f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                          f"MAE: {avg_train_mae:.4f}/{epoch_val_mae:.4f} | "
                          f"SNR: {epoch_val_snr:.2f}dB | P-MAE: {epoch_val_pmae:.4f}")


                    if epoch_val_mae < best_val_mae:
                        best_val_mae = epoch_val_mae
                        best_model_state = copy.deepcopy(model.state_dict())
                else:
                    print(f"Epoch [{epoch + 1}/{epochs}] Done. Train Loss: {avg_train_loss:.4f}")


        except KeyboardInterrupt:
            print("\nTraining interrupted. Finalizing...")

        if save_model and model_path and best_model_state is not None:
            torch.save(best_model_state, model_path)

            print(f"⭐ Best model saved (MAE: {best_val_mae:.4f})")

        return history, best_model_state

    def validate(self, model, val_loader, criterion, device='cpu'):


        model.eval()
        model.to(device)

        val_batch_metrics = []
        total_val_loss = 0


        pbar = tqdm(val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                input_tensor = batch['data'].to(device, non_blocking=True)
                meta_tensor = batch['demo'].to(device, non_blocking=True)


                if self.embed_mode == 'bioembed':
                    phase_tensor = batch['phase'].to(device, non_blocking=True)
                    output_tensor, mask = model(input_tensor, meta_tensor, phase_tensor)
                else:
                    output_tensor, mask = model(input_tensor, meta_tensor)


                loss_dict = criterion(output_tensor, input_tensor, mask=mask)
                total_val_loss += loss_dict['loss'].item()


                metrics = compute_metrics(input_tensor, output_tensor, mask)
                val_batch_metrics.append(metrics)


        num_batches = len(val_loader)
        avg_metrics = {
            'val_loss': total_val_loss / num_batches,
            'val_mae': np.mean([m['mae'] for m in val_batch_metrics]),
            'val_mse': np.mean([m['mse'] for m in val_batch_metrics]),
            'val_snr': np.mean([m['snr'] for m in val_batch_metrics]),
            'val_pmae': np.mean([m['pmae'] for m in val_batch_metrics]),
            'val_r2': np.mean([m['r2'] for m in val_batch_metrics])
        }

        return avg_metrics
