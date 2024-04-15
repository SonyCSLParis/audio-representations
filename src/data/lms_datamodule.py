from typing import List, Tuple

from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from src.data.components.lms_dataset import build_dataset


class LMSDataModule(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 dataset: str,
                 crop_frames: int,
                 norm_stats: Tuple[float, float] | None = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 devices: int | List[int] = 1):
        super().__init__()
        if not isinstance(devices, int):
            devices = len(devices)

        self.dataset_kwargs = dict(
            data_path=data_path,
            dataset_name=dataset,
            crop_frames=crop_frames,
            norm_stats=norm_stats
        )
        self.dataloader_kwargs = dict(
            batch_size=batch_size // devices,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.dataset = None

    def setup(self, stage):
        self.dataset = build_dataset(**self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=True, **self.dataloader_kwargs)
