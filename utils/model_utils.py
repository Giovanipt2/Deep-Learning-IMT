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
    input_data_or_size: Union[torch.Tensor, tuple[int, ...], torch.Size],
    device: Optional[Union[torch.device, str]] = None,
) -> float:
    """
    Computes estimated GFLOPs for a single forward pass of a PyTorch model using `torchinfo`.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        input_data_or_size (Union[torch.Tensor, tuple[int, ...], torch.Size]):
            Either a sample input Tensor or a shape tuple including batch size (e.g., (1, 1, 28, 28)).
        device (Optional[Union[torch.device, str]]): Target device for evaluation pass.

    Returns:
        float: Estimated total GFLOPs. Returns 0.0 if computation fails.
    """
    try:
        was_training = model.training
        model.eval()

        # If it is a Tensor, ensure it's just 1 batch sample to measure 1 pass
        if isinstance(input_data_or_size, torch.Tensor):
            sample_input = input_data_or_size[:1] if input_data_or_size.ndim > 0 else input_data_or_size
            stats = summary(model, input_data=sample_input, device=device, verbose=0)
        else:
            stats = summary(model, input_size=input_data_or_size, device=device, verbose=0)

        model.train(was_training)

        # 1 MAC (Multiply-Accumulate) = 2 FLOPs
        total_macs = stats.total_mult_adds
        total_flops = total_macs * 2

        return total_flops / 1e9

    except Exception as e:
        logger.warning(f"Failed to compute GFLOPs using torchinfo: {e}")
        return 0.0
