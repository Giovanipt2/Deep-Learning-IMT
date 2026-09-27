"""
Checkpoint management utility module.

Provides a robust interface for saving, loading, and resuming PyTorch model training states,
ensuring seamless persistence of model weights, optimizer states, and training metadata.
"""

import logging
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and restoring PyTorch model checkpoints.

    Saves both the latest training state ('last_checkpoint.pt') for session resumption
    and the optimal performing state ('best_checkpoint.pt') based on validation metrics.

    Attributes:
        checkpoint_dir (Path): Directory where checkpoint files are stored.
        last_checkpoint_path (Path): Path to the most recent checkpoint file.
        best_checkpoint_path (Path): Path to the best performing checkpoint file.
    """
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        file_prefix: str = "",
    ) -> None:
        """
        Initializes the CheckpointManager.

        Args:
            checkpoint_dir (str | Path): Directory path where checkpoint files will be stored.
            file_prefix (str): Optional prefix for checkpoint filenames. Defaults to "".
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
        additional_state: dict[str, Any] | None = None,
    ) -> None:
        """
        Saves the current training state to disk.

        Args:
            epoch (int): Current epoch index (0-based or 1-based).
            model (nn.Module): PyTorch model instance whose weights will be saved.
            optimizer (optim.Optimizer): PyTorch optimizer instance whose state will be saved.
            metric_value (float): Evaluation metric score associated with this state.
            is_best (bool): If True, also updates the best checkpoint file. Defaults to False.
            additional_state (dict[str, Any] | None): Optional extra states to persist
                (e.g., learning rate schedulers, custom metrics). Defaults to None.

        Returns:
            None
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        }

        if additional_state is not None:
            state.update(additional_state)

        # Always persist the latest checkpoint for training session resumption
        torch.save(state, self.last_checkpoint_path)
        logger.info(f"Latest checkpoint saved to: {self.last_checkpoint_path}")

        # Update the best performing checkpoint file if requested
        if is_best:
            torch.save(state, self.best_checkpoint_path)
            logger.info(f"Best model checkpoint updated at: {self.best_checkpoint_path}")


    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
        load_best: bool = False,
    ) -> dict[str, Any]:
        """
        Loads a saved checkpoint into the model and optional optimizer.

        Args:
            model (nn.Module): PyTorch model instance to receive the saved weights.
            optimizer (optim.Optimizer | None): Optional optimizer instance to restore state into.
                Defaults to None.
            device (torch.device | str): Target compute device for mapping loaded tensors.
                Defaults to "cpu".
            load_best (bool): If True, loads 'best_checkpoint.pt'; otherwise loads
                'last_checkpoint.pt'. Defaults to False.

        Returns:
            dict[str, Any]: Metadata dictionary containing epoch and metric information,
                excluding state dicts.

        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        target_path = self.best_checkpoint_path if load_best else self.last_checkpoint_path

        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {target_path}")

        logger.info(f"Loading checkpoint from: {target_path}")
        checkpoint = torch.load(target_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Return metadata dictionary without the heavy parameter tensors
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
            load_best (bool): If True, checks for 'best_checkpoint.pt'; otherwise
                checks for 'last_checkpoint.pt'. Defaults to False.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        target_path = self.best_checkpoint_path if load_best else self.last_checkpoint_path
        return target_path.exists()
