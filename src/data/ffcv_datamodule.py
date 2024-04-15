from typing import List, Tuple, Any

import torch

from lightning.pytorch import LightningDataModule

from src.data.components.ffcv_audio import build_dataloader


class FFCVDataModule(LightningDataModule):
    def __init__(self,
                 data_path: str,
                 input_size: Tuple[int, int],
                 norm_stats: Tuple[float, float] | None = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 seed: int = None,
                 devices: int | List[int] = 1):
        super().__init__()
        if not isinstance(devices, int):
            devices = len(devices)

        self.loader_kwargs = dict(
            data_path=data_path,
            input_size=input_size,
            norm_stats=norm_stats,
            batch_size=batch_size // devices,
            num_workers=num_workers,
            distributed=True if devices > 1 else False,
            seed=seed
        )

    def train_dataloader(self):
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        return build_dataloader(**self.loader_kwargs, device=device)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        r"""This part is handled by FFCV so we disable Lightning's one."""
        return batch
