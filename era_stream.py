from typing import Dict, Optional, Callable

import torch
import xarray as xr
import xbatcher
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    """Implements a PyTorch Dataset for ERA5 data stored in an xarray Dataset.
        Class supports torchvision transforms.
    """
    def __init__(
        self,
        dataset: xr.Dataset,
        patch_size: Dict[str, int],
        transform: Optional[Callable] = None,
        overlap: Optional[Dict[str, int]] = None,
    ):
        self.dataset = dataset
        self.patch_size = patch_size
        self.transform = transform
        self.overlap = overlap or {}

        # Create batch generator
        self.batch_gen = xbatcher.BatchGenerator(
            self.dataset,
            input_dims=patch_size,
            preload_batch=False,
        )

    def __len__(self):
        return len(self.batch_gen)

    def __getitem__(self, idx):
        batch = self.batch_gen[idx]
        if self.transform:
            batch = self.transform(batch)
        return torch.tensor(batch.to_array().values, dtype=torch.float32)
