"""
Checkpoint management utility module.

Provides a robust interface for saving, loading, and resuming PyTorch model training states,
ensuring seamless persistence of model weights, optimizer states, and training metadata.
"""

import logging
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and restoring PyTorch model checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        file_prefix: str = "",
    ) -> None:
        """
        Initializes the CheckpointManager.

        Args:
            checkpoint_dir (str | Path): Directory where checkpoints will be saved.
            file_prefix (str): Optional prefix for checkpoint filenames.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{file_prefix}_" if file_prefix else ""
        self.last_checkpoint_path = self.checkpoint_dir / f"{prefix}last_checkpoint.pt"
        self.best_checkpoint_path = self.checkpoint_dir / f"{prefix}best_checkpoint.pt"


    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metric_value: float,
        is_best: bool = False,
        additional_state: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Saves current training state to disk.

        Args:
            epoch (int): Current training epoch.
            model (nn.Module): The PyTorch model to save.
            optimizer (optim.Optimizer): The optimizer to save.
            metric_value (float): The metric value used to determine if this is the best checkpoint.
            is_best (bool): If True, saves this checkpoint as the best performing model.
            additional_state (Optional[dict[str, Any]]): Any additional state information to save.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        }

        if additional_state is not None:
            state.update(additional_state)

        # Always persist the latest checkpoint
        torch.save(state, self.last_checkpoint_path)

        # Update the best performing checkpoint file if requested
        if is_best:
            torch.save(state, self.best_checkpoint_path)
            logger.info(f"New best model checkpoint saved (Metric: {metric_value:.4f}) at: {self.best_checkpoint_path}")


    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: torch.device | str = "cpu",
        load_best: bool = True,
    ) -> dict[str, Any]:
        """
        Loads a saved checkpoint into the model and optional optimizer.

        Args:
            model (nn.Module): The PyTorch model to load the state into.
            optimizer (Optional[optim.Optimizer]): The optimizer to load the state into.
            device (torch.device | str): The device to map the loaded checkpoint to.
            load_best (bool): If True, loads the best checkpoint; otherwise, loads the last
            checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the checkpoint metadata (e.g., epoch, metric_value).
        """
        target_path = self.best_checkpoint_path if load_best else self.last_checkpoint_path

        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {target_path}")

        logger.info(f"Loading checkpoint from: {target_path}")
        checkpoint = torch.load(target_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in ("model_state_dict", "optimizer_state_dict")
        }
        return metadata


    def has_checkpoint(self, load_best: bool = False) -> bool:
        """
        Checks whether a specified checkpoint file exists on disk.

        Args:
            load_best (bool): If True, checks for the best checkpoint; otherwise, checks for
            the last checkpoint.

        Returns:
            bool: True if the specified checkpoint file exists, False otherwise.
        """
        target_path = self.best_checkpoint_path if load_best else self.last_checkpoint_path
        return target_path.exists()
