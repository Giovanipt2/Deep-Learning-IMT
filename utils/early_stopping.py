"""
Early stopping utility module.

Monitors a specified validation metric during training and signals when to stop
training early to prevent overfitting and avoid unnecessary computation.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Tracks validation metrics and determines if training should stop early.

    Monitors whether a given performance metric (e.g., validation loss or accuracy)
    has stopped improving after a specified number of consecutive epochs ('patience').

    Attributes:
        patience (int): Number of epochs to wait after the last metric improvement.
        min_delta (float): Minimum absolute change to qualify as an improvement.
        mode (str): Optimization objective ('min' for loss, 'max' for accuracy, F1, etc).
        counter (int): Number of consecutive epochs without metric improvement.
        best_score (float | None): Best metric value recorded so far.
        early_stop (bool): Flag indicating whether early stopping criterion is met.
        is_best (bool): Flag indicating whether the current score is the best so far.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int): Number of epochs with no improvement after which training
                will be stopped. Defaults to 7.
            min_delta (float): Minimum absolute change in the monitored metric to qualify
                as an improvement. Defaults to 0.0.
            mode (str): Metric optimization objective. Options: 'min' or 'max'.
                Defaults to 'min'.

        Raises:
            ValueError: If patience < 1, min_delta < 0.0, or mode is not 'min' or 'max'.
        """
        if patience < 1:
            raise ValueError("Patience must be a positive integer greater than or equal to 1.")
        if min_delta < 0.0:
            raise ValueError("min_delta must be greater than or equal to 0.0.")
        if mode not in ("min", "max"):
            raise ValueError("Mode must be either 'min' or 'max'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter: int = 0
        self.best_score: float | None = None
        self.early_stop: bool = False
        self.is_best: bool = False


    def __call__(self, metric_value: float) -> bool:
        """
        Evaluates the current metric value and updates internal tracking state.

        Args:
            metric_value (float): Current epoch's evaluation metric score.

        Returns:
            bool: True if early stopping criterion is met, False otherwise.
        """
        if self.best_score is None:
            self.best_score = metric_value
            self.is_best = True
            logger.info(f"Initial metric score established: {metric_value:.6f}")
            return False

        has_improved = self._check_improvement(metric_value)

        if has_improved:
            logger.info(
                f"Metric improved from {self.best_score:.6f} to {metric_value:.6f}. "
                f"Resetting patience counter (0/{self.patience})."
            )
            self.best_score = metric_value
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
            logger.info(
                f"No metric improvement for {self.counter}/{self.patience} epochs. "
                f"Best score: {self.best_score:.6f}"
            )

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping threshold reached. Triggering training stop.")

        return self.early_stop


    def _check_improvement(self, current_score: float) -> bool:
        """
        Evaluates whether current_score improves upon best_score according to mode and min_delta.

        Args:
            current_score (float): Current metric score to evaluate.

        Returns:
            bool: True if the score improved significantly, False otherwise.
        """
        if self.best_score is None:
            return True

        if self.mode == "min":
            # For loss minimization: score must decrease by at least min_delta
            return current_score < (self.best_score - self.min_delta)

        # For accuracy maximization: score must increase by at least min_delta
        return current_score > (self.best_score + self.min_delta)


    def reset(self) -> None:
        """
        Resets the internal tracking state.
        Allows re-using the same EarlyStopping instance across multiple training runs.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_best = False
        logger.info("EarlyStopping state has been reset.")
