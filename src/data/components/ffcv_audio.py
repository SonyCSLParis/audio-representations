from typing import Tuple

import torch
from torchvision.transforms import Normalize

from ffcv.fields import SpectrogramField
from ffcv.fields.decoders import SpectrogramDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor


def build_dataloader(data_path: str,
                     input_size: Tuple[int, int],
                     norm_stats: Tuple[float, float] | None = None,
                     batch_size: int = 32,
                     num_workers: int = 0,
                     device: torch.device = torch.device("cpu"),
                     distributed: bool = False,
                     seed: int = None) -> Loader:
    # parse config
    n_mels, n_frames = input_size

    # define pipeline
    pipeline = [SpectrogramDecoder(n_frames), ToTensor(), ToDevice(device)]
    if norm_stats:
        norm_stats = torch.tensor(norm_stats)
        normalize = Normalize(*norm_stats)
        pipeline.append(normalize)

    # create dataloader
    return Loader(
        data_path + ".beton",
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        distributed=distributed,
        seed=seed,
        pipelines={"audio": pipeline},
        custom_fields={"audio": SpectrogramField}
    )
