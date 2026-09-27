"""
Device management and reproducibility utility module.

Provides functions for detecting hardware accelerators (Apple Silicon MPS, NVIDIA CUDA,
or CPU execution) and setting global random seeds across libraries for experiment reproducibility.
"""

import logging
import os
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Detects and returns the optimal available compute device.

    Prioritizes Apple Silicon (MPS) if `prefer_mps` is True, followed by NVIDIA (CUDA).
    Falls back to CPU execution if no hardware accelerator is available.

    Args:
        prefer_mps (bool): Whether to prioritize Metal Performance Shaders (MPS)
            on Apple Silicon Macs. Defaults to True.

    Returns:
        torch.device: PyTorch device object representing the selected compute hardware.

    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='mps')
    """
    if prefer_mps and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning(
                "MPS is available on this platform, but PyTorch was not built with MPS support. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

        # Enable automatic CPU fallback for PyTorch operations not yet native to MPS
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is None:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        logger.info("Selected hardware accelerator: Apple Silicon GPU (MPS)")
        return torch.device("mps")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Selected hardware accelerator: NVIDIA GPU ({device_name})")
        return torch.device("cuda")

    logger.info("No GPU accelerator detected. Defaulting to CPU execution.")
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic_gpus: bool = True) -> None:
    """
    Sets global seeds for random number generators across Python, NumPy, and PyTorch.

    Args:
        seed (int): The integer seed value. Defaults to 42.
        deterministic_gpus (bool): If True, configures PyTorch backends to enforce
            deterministic CUDA execution. Defaults to True.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_gpus:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.info(f"Global seed fixed to: {seed}")
