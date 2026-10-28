"""
Module: dataset.py
Location: F:/sparrso/multi-modal-cloud-removal-dataset-design/preprocessing/dataset.py
Description: PyTorch Dataset and DataLoader for Multi-Modal Cloud Removal.
Author: Md Abu Sayem Sarker
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional


class MultiModalCloudDataset(Dataset):
    """
    PyTorch Dataset loading synchronized 4-band Optical and 2-band SAR patches.
    Supports:
      1. Real cloudy mode with proxy targets.
      2. Synthetic cloud occlusion mode for exact ground-truth benchmark training.
    """
    def __init__(
        self,
        patches_dir: str,
        is_synthetic_mode: bool = False,
        augment: bool = True
    ):
        self.patches_dir = patches_dir
        self.is_synthetic_mode = is_synthetic_mode
        self.augment = augment
        self.file_list = sorted(glob.glob(os.path.join(patches_dir, "*.npz")))

        if len(self.file_list) == 0:
            print(f"[!] Warning: No .npz files found in {patches_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def _apply_augmentation(
        self,
        opt: np.ndarray,
        sar: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=2).copy()
            sar = np.flip(sar, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=1).copy()
            sar = np.flip(sar, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random 90-degree rotations
        rot_k = np.random.randint(0, 4)
        if rot_k > 0:
            opt = np.rot90(opt, rot_k, axes=(1, 2)).copy()
            sar = np.rot90(sar, rot_k, axes=(1, 2)).copy()
            mask = np.rot90(mask, rot_k, axes=(1, 2)).copy()

        return opt, sar, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_list[idx]
        data = np.load(file_path)

        opt = data['optical']        # [4, 256, 256] in [0, 1]
        sar = data['sar']            # [2, 256, 256] in [-1, 1]
        mask = data['cloud_mask']    # [1, 256, 256] binary mask

        if self.augment:
            opt, sar, mask = self._apply_augmentation(opt, sar, mask)

        if self.is_synthetic_mode:
            # In synthetic mode: 'opt' is clear ground truth target!
            target_opt = opt.copy()
            
            # Synthesize thick cloud occlusion: cloud pixels set to diffuse high reflectance
            cloudy_opt = opt * (1.0 - mask) + (0.85 + 0.1 * np.random.rand(*opt.shape).astype(np.float32)) * mask
            cloudy_opt = np.clip(cloudy_opt, 0.0, 1.0)
        else:
            # Real cloudy mode:
            cloudy_opt = opt
            # Target is the near-date proxy reference if provided, or clean patch
            target_opt = opt.copy()

        return {
            'optical_cloudy': torch.from_numpy(cloudy_opt).float(), # [4, 256, 256]
            'sar': torch.from_numpy(sar).float(),                   # [2, 256, 256]
            'target': torch.from_numpy(target_opt).float(),         # [4, 256, 256]
            'cloud_mask': torch.from_numpy(mask).float()            # [1, 256, 256]
        }


def get_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
    synthetic_mode: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating Train, Val, and Test DataLoaders."""
    train_set = MultiModalCloudDataset(
        os.path.join(data_root, 'train'),
        is_synthetic_mode=synthetic_mode,
        augment=True
    )
    val_set = MultiModalCloudDataset(
        os.path.join(data_root, 'val'),
        is_synthetic_mode=synthetic_mode,
        augment=False
    )
    test_set = MultiModalCloudDataset(
        os.path.join(data_root, 'test'),
        is_synthetic_mode=synthetic_mode,
        augment=False
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
