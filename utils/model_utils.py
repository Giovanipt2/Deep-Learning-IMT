"""
Model utility functions for computing complexity metrics, parameter counts, and FLOPs.
"""

import logging
from typing import Optional, Union
import torch
import torch.nn as nn
from torchinfo import summary

logger = logging.getLogger(__name__)


def get_model_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Calculates the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        trainable_only (bool): If True, counts only parameters requiring gradients.
                              If False, counts all parameters. Defaults to True.

    Returns:
        int: Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_gflops(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 1, 28, 28),
    device: Optional[Union[torch.device, str]] = None,
) -> float:
    """
    Computes the estimated GFLOPs (Giga Floating Point Operations) for a single forward
    pass of a PyTorch model given an input shape, using `torchinfo`.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        input_size (tuple[int, ...]): Input tensor shape including batch size, e.g., (1, 1, 28, 28).
                                       Defaults to (1, 1, 28, 28).
        device (Optional[Union[torch.device, str]]): Target device for evaluation pass.
                                                    If None, uses CPU or the model's active device.

    Returns:
        float: Estimated total GFLOPs. Returns 0.0 if computation fails.
    """
    try:
        was_training = model.training
        model.eval()

        stats = summary(
            model,
            input_size=input_size,
            device=device,
            verbose=0,
        )

        model.train(was_training)

        # total_mult_adds represents Multiply-Accumulate Operations (MACs).
        # Standard convention: 1 MAC = 2 FLOPs (1 multiplication + 1 addition).
        total_macs = stats.total_mult_adds
        total_flops = total_macs * 2

        return total_flops / 1e9

    except Exception as e:
        logger.warning(f"Failed to compute GFLOPs using torchinfo: {e}")
        return 0.0
