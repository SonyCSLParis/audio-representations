from typing import Tuple

import torch


def unstructured_mask(m: int,
                        n: int,
                        masking_ratio: float,
                        num_masks: int = 1,
                        device: torch.device = torch.device("cpu")) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    p = round((1 - masking_ratio) * m * n)

    flat_masks = torch.zeros((num_masks, m * n), dtype=torch.bool, device=device)

    # Generate random noise for all masks
    noise = torch.rand((num_masks, m * n), device=device)

    # Sort the noise tensor along the last dimension
    positions = noise.argsort(dim=-1)

    # Get the first p positions for each mask
    positions = positions[:, :p]

    # Set the corresponding elements to True in the flat_masks
    flat_masks.scatter_(1, positions, True)

    masks = flat_masks.view(num_masks, m, n)

    return masks, ~masks
