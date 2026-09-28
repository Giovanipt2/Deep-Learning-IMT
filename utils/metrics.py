"""
Metric tracking and evaluation utility module.

Provides functional metrics for classification and regression tasks, along with a flexible
MetricTracker class capable of tracking running scalar averages (e.g., loss) and accumulating
epoch-level predictions for exact non-linear metric computation.
"""

import logging
from typing import Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Functional Classification Metrics
# =============================================================================

def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculates classification accuracy score.

    Args:
        y_pred (torch.Tensor): Predicted class indices (shape: [N]) or raw logits (shape: [N, C]).
        y_true (torch.Tensor): Ground truth target class labels (shape: [N]).

    Returns:
        float: Accuracy score in range [0.0, 1.0].
    """
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    return float(correct / total) if total > 0 else 0.0


def calculate_precision_recall_f1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> dict[str, float]:
    """
    Computes Precision, Recall, and F1-Score for multiclass classification.

    Args:
        y_pred (torch.Tensor): Predicted class labels (shape: [N]) or logits (shape: [N, C]).
        y_true (torch.Tensor): Target class labels (shape: [N]).
        num_classes (int): Total number of unique classes.
        average (str): Aggregation mode ('macro', 'micro', or 'weighted'). Defaults to 'macro'.
            - 'macro': Unweighted mean across classes. Treats all classes equally regardless of support.
            - 'micro': Global aggregation. Calculates metrics from total true positives/false counts.
            - 'weighted': Support-weighted mean. Scales each class score by its proportion of real targets.

    Returns:
        dict[str, float]: Dictionary containing 'precision', 'recall', and 'f1_score'.
    """
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # Calculate confusion matrix entries per class
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    support = np.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = np.sum((y_pred_np == c) & (y_true_np == c))
        fp[c] = np.sum((y_pred_np == c) & (y_true_np != c))
        fn[c] = np.sum((y_pred_np != c) & (y_true_np == c))
        support[c] = np.sum(y_true_np == c)

    if average == "micro":
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        # Per-class metrics calculation
        precision_per_class = np.zeros(num_classes)
        recall_per_class = np.zeros(num_classes)
        f1_per_class = np.zeros(num_classes)

        for c in range(num_classes):
            precision_per_class[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            recall_per_class[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            p, r = precision_per_class[c], recall_per_class[c]
            f1_per_class[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if average == "weighted":
            weights = support / np.sum(support) if np.sum(support) > 0 else np.zeros(num_classes)
            precision = float(np.sum(precision_per_class * weights))
            recall = float(np.sum(recall_per_class * weights))
            f1 = float(np.sum(f1_per_class * weights))
        else:  # macro
            precision = float(np.mean(precision_per_class))
            recall = float(np.mean(recall_per_class))
            f1 = float(np.mean(f1_per_class))

    return {"precision": precision, "recall": recall, "f1_score": f1}


def calculate_roc_auc(
    y_prob: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Args:
        y_prob (torch.Tensor): Predicted probabilities or logits (shape: [N, C] or [N]).
        y_true (torch.Tensor): Ground truth target class labels (shape: [N]).
        num_classes (int): Total number of unique classes.

    Returns:
        float: One-vs-Rest macro-averaged ROC-AUC score.
    """
    if y_prob.ndim == 1 or num_classes == 2:
        probs = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        probs_np = probs.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        return _binary_roc_auc(probs_np, y_true_np)

    # Multiclass One-vs-Rest calculation
    probs_np = torch.softmax(y_prob, dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    auc_scores = []
    for c in range(num_classes):
        binary_y_true = (y_true_np == c).astype(int)
        binary_probs = probs_np[:, c]
        if len(np.unique(binary_y_true)) > 1:
            auc_scores.append(_binary_roc_auc(binary_probs, binary_y_true))

    return float(np.mean(auc_scores)) if len(auc_scores) > 0 else 0.0


def _binary_roc_auc(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """
    Helper function to calculate binary ROC-AUC via rank summation.

    Args:
        y_prob (torch.Tensor): Predicted probabilities or logits (shape: [N, C] or [N]).
        y_true (torch.Tensor): Ground truth target class labels (shape: [N]).
        num_classes (int): Total number of unique classes.

    Returns:
        float: Binary ROC-AUC score.
    """
    desc_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    rank = np.arange(len(y_true), 0, -1)
    rank_pos = np.sum(rank[y_true_sorted == 1])

    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calculate_cohen_kappa(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    weights: str | None = None,
) -> float:
    """
    Computes Cohen's Kappa score for inter-rater agreement.

    Args:
        y_pred (torch.Tensor): Predicted class labels or logits (shape: [N] or [N, C]).
        y_true (torch.Tensor): Target class labels (shape: [N]).
        num_classes (int): Total number of unique classes.
        weights (str | None): Weighting type: None (unweighted), 'linear', or 'quadratic' (QWK).
            Defaults to None.
            - None: Unweighted. Penalizes all misclassifications equally. Best for nominal
                    classes without inherent order (e.g., MNIST digits, CIFAR-10 objects).
            - 'linear': Linearly weighted. Penalty grows proportionally with distance (|i - j|).
                    Best for ordinal classes with linear distance steps (e.g., star ratings 1-5).
            - 'quadratic': Quadratically weighted (QWK). Penalty grows with squared distance ((i - j)²).
                    Heavily penalizes extreme errors; best for ordinal/graded domains (e.g., medical severity stages, essay grading).

    Returns:
        float: Cohen's Kappa score in range [-1.0, 1.0].
    """
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # Build confusion matrix O
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true_np, y_pred_np):
        O[t, p] += 1.0

    N = np.sum(O)
    if N == 0:
        return 0.0

    # Expected matrix E
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    E = np.outer(hist_true, hist_pred) / N

    if weights is None:
        W = np.ones((num_classes, num_classes)) - np.eye(num_classes)
    else:
        w_mat = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                diff = abs(i - j)
                if weights == "linear":
                    w_mat[i, j] = diff / (num_classes - 1)
                elif weights == "quadratic":
                    w_mat[i, j] = (diff / (num_classes - 1)) ** 2
                else:
                    raise ValueError("Weights mode must be None, 'linear', or 'quadratic'.")
        W = w_mat

    num = np.sum(W * O)
    den = np.sum(W * E)

    return float(1.0 - (num / den)) if den > 0 else 0.0


# =============================================================================
# Functional Regression Metrics
# =============================================================================

def calculate_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Mean Squared Error (MSE).

    Args:
        y_pred (torch.Tensor): Predicted continuous values (shape: [N]).
        y_true (torch.Tensor): Ground truth continuous values (shape: [N]).

    Returns:
        float: Mean Squared Error value.
    """
    return float(torch.mean((y_pred - y_true) ** 2).item())


def calculate_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Mean Absolute Error (MAE).

    Args:
        y_pred (torch.Tensor): Predicted continuous values (shape: [N]).
        y_true (torch.Tensor): Ground truth continuous values (shape: [N]).

    Returns:
        float: Mean Absolute Error value.
    """
    return float(torch.mean(torch.abs(y_pred - y_true)).item())


def calculate_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Root Mean Squared Error (RMSE).

    Args:
        y_pred (torch.Tensor): Predicted continuous values (shape: [N]).
        y_true (torch.Tensor): Ground truth continuous values (shape: [N]).

    Returns:
        float: Root Mean Squared Error value.
    """
    return float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item())


def calculate_r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes Coefficient of Determination (R² Score).

    Args:
        y_pred (torch.Tensor): Predicted continuous values (shape: [N]).
        y_true (torch.Tensor): Ground truth continuous values (shape: [N]).

    Returns:
        float: R² score value in range (-∞, 1.0].
    """
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


# =============================================================================
# MetricTracker Class
# =============================================================================

class MetricTracker:
    """
    Flexible tracker for managing running scalar metrics and accumulating predictions.

    Supports lightweight running averages (for loss) and batch-wise tensor accumulation
    for exact epoch-level evaluation of complex metrics (Accuracy, F1, ROC-AUC, Kappa).
    """

    def __init__(self) -> None:
        """Initializes empty metric containers."""
        self.scalar_sums: dict[str, float] = {}
        self.scalar_counts: dict[str, int] = {}

        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []


    def update_scalar(self, name: str, value: float, n: int = 1) -> None:
        """
        Updates a running scalar metric (e.g., loss).

        Args:
            name (str): Identifier name for the metric.
            value (float): Metric value for the current batch.
            n (int): Batch size / weight for weighted average. Defaults to 1.
        """
        self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + (value * n)
        self.scalar_counts[name] = self.scalar_counts.get(name, 0) + n


    def update_predictions(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Accumulates batch predictions and targets for epoch-level evaluation.

        Tensors are detached and stored on CPU to prevent GPU memory leaks.

        Args:
            y_pred (torch.Tensor): Model output predictions or logits.
            y_true (torch.Tensor): Ground truth target labels.
        """
        self._predictions.append(y_pred.detach().cpu())
        self._targets.append(y_true.detach().cpu())


    def get_scalar_average(self, name: str) -> float:
        """
        Returns the running average for a specified scalar metric.

        Args:
            name (str): Identifier name for the metric.

        Returns:
            float: Current average value of the metric, or 0.0 if not updated yet.
        """
        count = self.scalar_counts.get(name, 0)
        return self.scalar_sums.get(name, 0.0) / count if count > 0 else 0.0


    def get_accumulated_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates and returns all accumulated batch predictions and targets.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Concatenated predictions and targets tensors.
        """
        if not self._predictions or not self._targets:
            raise RuntimeError("No predictions or targets have been accumulated.")

        all_preds = torch.cat(self._predictions, dim=0)
        all_targets = torch.cat(self._targets, dim=0)
        return all_preds, all_targets


    def compute_classification_summary(
        self,
        num_classes: int,
        kappa_weights: str | None = None,
    ) -> dict[str, float]:
        """
        Calculates all accumulated classification metrics for the epoch.

        Args:
            num_classes (int): Total number of unique dataset classes.
            kappa_weights (str | None): Cohen's Kappa weighting mode. Defaults to None.

        Returns:
            dict[str, float]: Dictionary containing computed scores for Accuracy,
                Precision, Recall, F1-Score, ROC-AUC, and Cohen's Kappa.
        """
        preds, targets = self.get_accumulated_tensors()

        acc = calculate_accuracy(preds, targets)
        prf1 = calculate_precision_recall_f1(preds, targets, num_classes=num_classes)
        auc = calculate_roc_auc(preds, targets, num_classes=num_classes)
        kappa = calculate_cohen_kappa(preds, targets, num_classes=num_classes, weights=kappa_weights)

        summary = {
            "accuracy": acc,
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1_score": prf1["f1_score"],
            "roc_auc": auc,
            "cohen_kappa": kappa,
        }
        return summary


    def compute_regression_summary(self) -> dict[str, float]:
        """
        Calculates all accumulated regression metrics for the epoch.

        Returns:
            dict[str, float]: Dictionary containing computed scores for MSE, MAE, RMSE, and R2 Score.
        """
        preds, targets = self.get_accumulated_tensors()

        mse = calculate_mse(preds, targets)
        mae = calculate_mae(preds, targets)
        rmse = calculate_rmse(preds, targets)
        r2 = calculate_r2_score(preds, targets)

        summary = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
        }
        return summary


    def reset(self) -> None:
        """Clears all accumulated scalars and tensor histories."""
        self.scalar_sums.clear()
        self.scalar_counts.clear()
        self._predictions.clear()
        self._targets.clear()
