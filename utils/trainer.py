"""
Model training and evaluation loop manager.

Provides a unified Trainer class responsible for running training and validation epochs,
managing hardware acceleration (AMP), gradient accumulation, gradient clipping, learning rate
scheduling, checkpointing, early stopping, and metric tracking across classification and regression tasks.
"""

import logging
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import CheckpointManager
from utils.device import get_device
from utils.early_stopping import EarlyStopping
from utils.metrics import MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Flexible engine for training and evaluating PyTorch models in classification or regression tasks.

    Handles mixed precision training, gradient clipping/accumulation, learning rate
    scheduling, metric history collection, checkpoint saving, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        early_stopping: Optional[EarlyStopping] = None,
        device: Optional[torch.device] = None,
        task_type: str = "classification",
        num_classes: Optional[int] = None,
        primary_metric: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        use_amp: bool = False,
        kappa_weights: Optional[str] = None,
    ) -> None:
        """
        Initializes the Trainer instance.

        Args:
            model (nn.Module): PyTorch neural network model.
            optimizer (optim.Optimizer): Optimization algorithm instance.
            criterion (nn.Module): Loss function.
            scheduler (Optional[Any]): Learning rate scheduler instance. Defaults to None.
            checkpoint_manager (Optional[CheckpointManager]): Manager for saving model state. Defaults to None.
            early_stopping (Optional[EarlyStopping]): Early stopping callback. Defaults to None.
            device (Optional[torch.device]): Compute device (CPU/CUDA/MPS). Auto-detected if None.
            task_type (str): Task mode ('classification' or 'regression'). Defaults to 'classification'.
            num_classes (Optional[int]): Total target classes. Required if task_type='classification'.
            primary_metric (Optional[str]): Main metric key to display in logs (e.g., 'f1_score', 'mae').
                Defaults to 'accuracy' for classification, 'rmse' for regression.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients before optimizer step. Defaults to 1.
            max_grad_norm (Optional[float]): Maximum norm for gradient clipping. Defaults to None.
            use_amp (bool): Whether to enable Automatic Mixed Precision (AMP). Defaults to False.
            kappa_weights (Optional[str]): Weighting scheme for Cohen's Kappa metric. Defaults to None.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be either 'classification' or 'regression'.")

        if task_type == "classification" and num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks.")

        self.task_type = task_type
        self.num_classes = num_classes
        self.primary_metric = primary_metric or ("accuracy" if task_type == "classification" else "rmse")

        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.kappa_weights = kappa_weights

        # Set up AMP Scaler (enabled if CUDA and requested)
        is_cuda = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and is_cuda))

        # Trackers for train and validation
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()

        # Training history repository
        self.history: dict[str, list[float]] = {}


    def _prepare_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """
        Extracts inputs and targets from a batch and transfers them to target device.

        Args:
            batch (Any): A single batch from DataLoader, can be tuple, list, or dict.

        Returns:
            tuple[Any, torch.Tensor]: Tuple of (inputs, targets) on the correct device.
        """
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x = batch.get("x", batch.get("inputs"))
            y = batch.get("y", batch.get("targets"))
        else:
            raise ValueError(f"Unsupported batch structure: {type(batch)}")

        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, (tuple, list)):
            x = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in x]

        y = y.to(self.device)
        return x, y


    def _compute_epoch_summary(self, tracker: MetricTracker) -> dict[str, float]:
        """
        Computes metrics summary dynamically based on task type.

        Args:
            tracker (MetricTracker): Tracker containing accumulated predictions and targets.

        Returns:
            dict[str, float]: Dictionary of computed metrics for the epoch.
        """
        if self.task_type == "classification":
            summary = tracker.compute_classification_summary(
                num_classes=self.num_classes, kappa_weights=self.kappa_weights
            )
        else:
            summary = tracker.compute_regression_summary()

        summary["loss"] = tracker.get_scalar_average("loss")
        return summary


    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        Runs a single training epoch across all batches.

        Args:
            train_loader (DataLoader): DataLoader for training dataset.
            epoch (int): Current epoch index.

        Returns:
            dict[str, float]: Dictionary of computed metrics for the epoch.
        """
        self.model.train()
        self.train_tracker.reset()
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for i, batch in enumerate(pbar):
            x, y = self._prepare_batch(batch)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                scaled_loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            # Optimizer step with accumulation and optional clipping
            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Record batch statistics
            self.train_tracker.update_scalar("loss", loss.item(), n=y.size(0))
            self.train_tracker.update_predictions(outputs, y)

            pbar.set_postfix({"loss": f"{self.train_tracker.get_scalar_average('loss'):.4f}"})

        return self._compute_epoch_summary(self.train_tracker)


    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        Runs validation evaluation across all batches without computing gradients.

        Args:
            val_loader (DataLoader): DataLoader for validation dataset.
            epoch (int): Current epoch index.

        Returns:
            dict[str, float]: Dictionary of computed metrics for the epoch.
        """
        self.model.eval()
        self.val_tracker.reset()

        pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
        with torch.no_grad():
            for batch in pbar:
                x, y = self._prepare_batch(batch)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                self.val_tracker.update_scalar("loss", loss.item(), n=y.size(0))
                self.val_tracker.update_predictions(outputs, y)

                pbar.set_postfix({"loss": f"{self.val_tracker.get_scalar_average('loss'):.4f}"})

        return self._compute_epoch_summary(self.val_tracker)


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
    ) -> dict[str, list[float]]:
        """
        Executes the full training and validation pipeline across specified epochs.

        Args:
            train_loader (DataLoader): DataLoader for training dataset.
            val_loader (DataLoader): DataLoader for validation dataset.
            epochs (int): Target total number of epochs to train.
            start_epoch (int): Epoch index to start/resume from. Defaults to 1.

        Returns:
            dict[str, list[float]]: Complete history dictionary tracking train and val metrics per epoch.
        """
        logger.info(f"Starting model training on device '{self.device}' for {epochs - start_epoch + 1} epochs.")

        try:
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = self._train_epoch(train_loader, epoch)
                val_metrics = self._validate_epoch(val_loader, epoch)

                # Append metrics to global history
                for key, val in train_metrics.items():
                    self.history.setdefault(f"train_{key}", []).append(val)
                for key, val in val_metrics.items():
                    self.history.setdefault(f"val_{key}", []).append(val)

                # Learning rate scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()

                # Dynamic logging based on primary_metric
                metric_name = self.primary_metric
                train_m_val = train_metrics.get(metric_name, 0.0)
                val_m_val = val_metrics.get(metric_name, 0.0)

                logger.info(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Train Loss: {train_metrics['loss']:.4f} - Train {metric_name.replace('_', ' ').title()}: {train_m_val:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} - Val {metric_name.replace('_', ' ').title()}: {val_m_val:.4f}"
                )

                # Save checkpoint if manager is configured
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        scheduler=self.scheduler,
                    )

                # Check early stopping criterion
                if self.early_stopping is not None:
                    self.early_stopping(val_metrics["loss"])
                    if self.early_stopping.early_stop:
                        logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break

        except KeyboardInterrupt:
            logger.warning("Training process interrupted manually by user. Returning accumulated history.")

        return self.history
