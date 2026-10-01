"""
Model training, evaluation, and persistence loop manager.
"""

import json
import logging
from pathlib import Path
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
from utils import get_model_gflops, get_model_parameters

logger = logging.getLogger(__name__)


class Trainer:
    """
    Engine for training, evaluating, and persisting PyTorch models.
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
        higher_is_better: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        use_amp: bool = False,
        kappa_weights: Optional[str] = None,
        model_name: str = "CustomModel",
        model_params: int = 0,
        model_gflops: float = 0.0,
    ) -> None:
        """
        Initializes the Trainer with model, optimizer, criterion, and optional components like scheduler,
        checkpoint manager, and early stopping.

        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (optim.Optimizer): The optimizer for training.
            criterion (nn.Module): The loss function.
            scheduler (Optional[Any]): Learning rate scheduler.
            checkpoint_manager (Optional[CheckpointManager]): Manages saving/loading checkpoints.
            early_stopping (Optional[EarlyStopping]): Early stopping mechanism.
            device (Optional[torch.device]): Device to run the model on. Defaults to GPU if available.
            task_type (str): Type of task: "classification" or "regression".
            num_classes (Optional[int]): Number of classes for classification tasks.
            primary_metric (Optional[str]): Metric to monitor for best model selection.
            higher_is_better (bool): Whether a higher value of the primary metric is better.
            gradient_accumulation_steps (int): Steps to accumulate gradients before updating weights.
            max_grad_norm (Optional[float]): Max norm for gradient clipping. None disables clipping.
            use_amp (bool): Whether to use Automatic Mixed Precision for training.
            kappa_weights (Optional[str]): Weights for Cohen's Kappa calculation in classification tasks.
            model_name (str): Name of the model for metadata tracking.
            model_params (int): Number of parameters in the model for metadata tracking.
            model_gflops (float): GFLOPs of the model for metadata tracking.

        Raises:
            ValueError: If task_type is not "classification" or "regression".
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be either 'classification' or 'regression'.")

        if task_type == "classification" and num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks.")

        self.task_type = task_type
        self.num_classes = num_classes
        self.primary_metric = primary_metric or ("accuracy" if task_type == "classification" else "rmse")

        # Determine metric direction (higher is better for accuracy/f1, lower for loss/rmse/mae)
        if primary_metric in ("loss", "rmse", "mse", "mae"):
            self.higher_is_better = False
        else:
            self.higher_is_better = higher_is_better

        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")

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

        # Metadata tracking for plot_model_comparison
        self.model_name = model_name

        # Extract model parameters and GFLOPS if not provided
        if model_params == 0:
            self.model_params = get_model_parameters(model)
        else:
            self.model_params = model_params

        self.model_gflops = model_gflops

        # Scaler for AMP
        is_cuda = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and is_cuda))

        # Trackers
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()
        self.history: dict[str, list[float]] = {}

        # Automatically write metadata.json if checkpoint_manager exists
        if self.checkpoint_manager is not None:
            self._save_metadata_json()

    def _save_metadata_json(self) -> None:
        """Saves static model metadata (name, params, gflops) to directory."""
        meta_path = self.checkpoint_manager.checkpoint_dir / "metadata.json"
        data = {
            "name": self.model_name,
            "params": self.model_params,
            "gflops": self.model_gflops,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


    def _save_history_json(self) -> None:
        """Saves complete epoch training history to JSON."""
        if self.checkpoint_manager is not None:
            hist_path = self.checkpoint_manager.checkpoint_dir / "history.json"
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)


    def _load_history_json(self) -> dict[str, list[float]]:
        """Loads persisted training history, if available."""
        if self.checkpoint_manager is None:
            return {}

        hist_path = self.checkpoint_manager.checkpoint_dir / "history.json"
        if not hist_path.exists():
            return {}

        with open(hist_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _restore_training_state(self, epochs: int, start_epoch: int) -> tuple[int, bool]:
        """Restores the latest complete checkpoint and reports whether training is finished."""
        if self.checkpoint_manager is None or not self.checkpoint_manager.has_checkpoint(load_best=False):
            return start_epoch, False

        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.model,
            optimizer=self.optimizer,
            device=self.device,
            load_best=False,
        )
        last_epoch = int(checkpoint.get("epoch", 0))
        self.history = self._load_history_json()

        for key, values in self.history.items():
            self.history[key] = values[:last_epoch]

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        early_stopping_state = checkpoint.get("early_stopping_state")
        if self.early_stopping is not None and early_stopping_state is not None:
            self.early_stopping.counter = early_stopping_state["counter"]
            self.early_stopping.best_score = early_stopping_state["best_score"]
            self.early_stopping.early_stop = early_stopping_state["early_stop"]
            self.early_stopping.is_best = early_stopping_state["is_best"]

        best_metric_value = checkpoint.get("metric_value")
        if self.checkpoint_manager.best_checkpoint_path.exists():
            best_checkpoint = torch.load(self.checkpoint_manager.best_checkpoint_path, map_location=self.device)
            best_metric_value = best_checkpoint.get("metric_value", best_metric_value)
        if best_metric_value is not None:
            self.best_metric_value = best_metric_value

        early_stopped = bool(checkpoint.get("early_stopped", False))
        training_complete = early_stopped or last_epoch >= epochs
        return last_epoch + 1, training_complete


    def _reset_training_state(self) -> None:
        """Resets in-memory state before a forced training run."""
        self.history = {}
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.train_tracker.reset()
        self.val_tracker.reset()
        if self.early_stopping is not None:
            self.early_stopping.reset()


    def _is_better(self, current: float) -> bool:
        """
        Checks if current metric value improves upon best recorded metric.

        Args:
            current (float): Current epoch's metric value.

        Returns:
            bool: True if current is better than best_metric_value, False otherwise.
        """
        if self.higher_is_better:
            return current > self.best_metric_value
        return current < self.best_metric_value


    def _prepare_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """
        Prepares a batch of data for model input, ensuring tensors are moved to the correct device.

        Args:
            batch (Any): A batch of data, which can be a tuple, list, or dict containing inputs and targets.

        Returns:
            tuple[Any, torch.Tensor]: A tuple containing the inputs and targets, both moved to the correct device.
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
        Computes a summary of metrics for the current epoch based on the provided MetricTracker.

        Args:
            tracker (MetricTracker): The MetricTracker containing accumulated metrics for the epoch.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics for the epoch.
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
        Executes a single training epoch, updating model weights and tracking metrics.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            epoch (int): Current epoch number.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics for the training epoch.
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

            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.train_tracker.update_scalar("loss", loss.item(), n=y.size(0))
            self.train_tracker.update_predictions(outputs, y)
            pbar.set_postfix({"loss": f"{self.train_tracker.get_scalar_average('loss'):.4f}"})

        return self._compute_epoch_summary(self.train_tracker)


    def _validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int,
        collect_inputs: bool = False,
        max_inputs: int = 15,
    ) -> dict[str, float]:
        """
        Executes a single validation epoch, evaluating model performance on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for the validation dataset.
            epoch (int): Current epoch number.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics for the validation epoch.
        """
        self.model.eval()
        self.val_tracker.reset()
        evaluation_inputs: list[torch.Tensor] = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
        with torch.no_grad():
            for batch in pbar:
                x, y = self._prepare_batch(batch)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                self.val_tracker.update_scalar("loss", loss.item(), n=y.size(0))
                self.val_tracker.update_predictions(outputs, y)
                if collect_inputs and isinstance(x, torch.Tensor) and sum(item.size(0) for item in evaluation_inputs) < max_inputs:
                    remaining = max_inputs - sum(item.size(0) for item in evaluation_inputs)
                    evaluation_inputs.append(x[:remaining].detach().cpu())
                pbar.set_postfix({"loss": f"{self.val_tracker.get_scalar_average('loss'):.4f}"})

        if collect_inputs:
            self._last_evaluation_inputs = torch.cat(evaluation_inputs, dim=0) if evaluation_inputs else None

        return self._compute_epoch_summary(self.val_tracker)


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        force_retrain: bool = False,
    ) -> dict[str, list[float]]:
        """
        Main training loop that iterates over epochs, performing training and validation,
            and managing checkpoints and early stopping.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Total number of epochs to train.
            start_epoch (int): Epoch number to start training from (useful for resuming).
            force_retrain (bool): If True, ignores existing checkpoints and starts a new run.

        Returns:
            dict[str, list[float]]: A dictionary containing the training history of metrics across epochs
        """
        if force_retrain:
            self._reset_training_state()
            start_epoch = 1
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.last_checkpoint_path.unlink(missing_ok=True)
                self.checkpoint_manager.best_checkpoint_path.unlink(missing_ok=True)
                (self.checkpoint_manager.checkpoint_dir / "history.json").unlink(missing_ok=True)
        else:
            start_epoch, training_complete = self._restore_training_state(epochs, start_epoch)
            if training_complete:
                logger.info("Training already completed. Restoring history without retraining.")
                return self.history

        if self.model_gflops == 0.0:
            try:
                # Gets the first batch from the train_loader to extract a sample tensor
                first_batch = next(iter(train_loader))
                x_sample, _ = self._prepare_batch(first_batch)

                # Calculates GFLOPs using the actual input tensor from the first batch
                self.model_gflops = get_model_gflops(
                    model=self.model,
                    input_data_or_size=x_sample[:1],
                    device=self.device
                )

                # If there is a checkpoint_manager, update the metadata.json with the computed GFLOPs
                if self.checkpoint_manager is not None:
                    self._save_metadata_json()
            except Exception as e:
                logger.warning(f"Could not automatically compute GFLOPs from train_loader: {e}")

        logger.info(f"Starting model training on device '{self.device}' for {epochs - start_epoch + 1} epochs.")

        try:
            for epoch in range(start_epoch, epochs + 1):
                train_metrics = self._train_epoch(train_loader, epoch)
                val_metrics = self._validate_epoch(val_loader, epoch)

                for key, val in train_metrics.items():
                    self.history.setdefault(f"train_{key}", []).append(val)
                for key, val in val_metrics.items():
                    self.history.setdefault(f"val_{key}", []).append(val)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()

                metric_name = self.primary_metric
                val_m_val = val_metrics.get(metric_name, 0.0)
                train_m_val = train_metrics.get(metric_name, 0.0)

                # Check if current epoch is the best performing
                is_best = self._is_better(val_m_val)
                if is_best:
                    self.best_metric_value = val_m_val

                logger.info(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Train Loss: {train_metrics['loss']:.4f} | Train {metric_name.title()}: {train_m_val:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val {metric_name.title()}: {val_m_val:.4f}"
                )

                early_stopped = False
                if self.early_stopping is not None:
                    if self.early_stopping.mode == "max":
                        target_eval_metric = val_metrics.get(self.primary_metric, val_metrics["loss"])
                    else:
                        target_eval_metric = val_metrics["loss"] if "loss" in val_metrics else val_metrics.get(self.primary_metric)

                    self.early_stopping(target_eval_metric)
                    if self.early_stopping.early_stop:
                        early_stopped = True
                        logger.info(f"Early stopping triggered at epoch {epoch}.")

                # Save the state only after the complete epoch and its stopping decision.
                if self.checkpoint_manager is not None:
                    sched_state = self.scheduler.state_dict() if self.scheduler else None
                    early_stopping_state = None
                    if self.early_stopping is not None:
                        early_stopping_state = {
                            "counter": self.early_stopping.counter,
                            "best_score": self.early_stopping.best_score,
                            "early_stop": self.early_stopping.early_stop,
                            "is_best": self.early_stopping.is_best,
                        }
                    self.checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        metric_value=val_m_val,
                        is_best=is_best,
                        additional_state={
                            "scheduler_state_dict": sched_state,
                            "val_metrics": val_metrics,
                            "target_epochs": epochs,
                            "early_stopped": early_stopped,
                            "training_complete": epoch >= epochs or early_stopped,
                            "early_stopping_state": early_stopping_state,
                        },
                    )
                    self._save_history_json()

                if early_stopped:
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted manually. Saving accumulated history.")

        # Persist complete training history to JSON on disk
        self._save_history_json()
        return self.history


    def evaluate(self, test_loader: DataLoader, save_results: bool = True) -> dict[str, float]:
        """
        Runs evaluation on test dataset, optionally persisting 'eval_results.json' to disk.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            save_results (bool): Whether to save evaluation results to 'eval_results.json'.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics for the test dataset.
        """
        logger.info("Starting model evaluation on test set.")
        test_metrics = self._validate_epoch(test_loader, epoch=0, collect_inputs=True)

        predictions, targets = self.val_tracker.get_accumulated_tensors()
        if self.checkpoint_manager is not None:
            predictions_path = self.checkpoint_manager.checkpoint_dir / "evaluation_predictions.pt"
            torch.save(
                {
                    "y_true": targets,
                    "y_pred": predictions,
                    "inputs": getattr(self, "_last_evaluation_inputs", None),
                    "task_type": self.task_type,
                    "num_classes": self.num_classes,
                },
                predictions_path,
            )
            logger.info(f"Evaluation predictions saved to: {predictions_path}")

        if save_results and self.checkpoint_manager is not None:
            eval_path = self.checkpoint_manager.checkpoint_dir / "eval_results.json"
            formatted_results = {f"test_{k}": v for k, v in test_metrics.items()}
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(formatted_results, f, indent=4)
            logger.info(f"Test evaluation results saved to: {eval_path}")

        return test_metrics
