"""
Data Visualization and Model Plotting Utilities.

Provides a comprehensive suite of visualization functions for Exploratory Data Analysis (EDA),
Dimensionality Reduction, Model Evaluation (Classification & Regression), and Explainable AI (XAI).
 Plotting functions save figures under ``figures/{dataset_name}`` or ``figures/{model_name}`` and close them after saving.
"""

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc


# =====================================================================
# Internal Helper Functions
# =====================================================================


def _to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """
    Converts PyTorch tensors, lists, or sequences into a NumPy array.

    Args:
        x (Union[torch.Tensor, np.ndarray, list]): Input data to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _format_scale(x: float, pos: Any = None) -> str:
    """
    Formats large parameter/FLOP numbers into human-readable K, M, B scales.

    Args:
        x (float): Value to format.
        pos (Any): Position (unused, required for FuncFormatter).

    Returns:
        str: Formatted string representation of the value.
    """
    if x >= 1e9:
        return f"{x * 1e-9:.1f}B"
    if x >= 1e6:
        return f"{x * 1e-6:.1f}M"
    if x >= 1e3:
        return f"{x * 1e-3:.1f}K"
    return str(int(x))


def _save_figure(fig: plt.Figure, output_name: str, filename: str) -> None:
    """
    Saves a figure in the standard ``figures/{output_name}`` directory,
    displays it once, and closes it to release resources.

    Args:
        fig (plt.Figure): Figure to save and display.
        output_name (str): Dataset or model name used as the output directory.
        filename (str): Filename to use for the saved figure.

    Returns:
        None: The figure is saved and displayed in place.
    """
    if not output_name or output_name in {"dataset", "model", "model_comparison"}:
        raise ValueError(
            "output_name must identify the dataset or model, for example "
            "'MNIST', 'CIFAR10', or 'mnist_simple_ffn'."
        )

    output_dir = Path("figures") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def load_evaluation_artifacts(
    model_name: str,
    checkpoint_root: Union[str, Path] = "checkpoints",
) -> Dict[str, Any]:
    """
    Loads persisted evaluation outputs and derives plot-ready predictions.

    Args:
        model_name (str): Name of the model whose evaluation artifacts are loaded.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.

    Returns:
        Dict[str, Any]: Dictionary containing targets, predictions, probabilities,
        optional inputs, task type, and number of classes.
    """
    artifact_path = Path(checkpoint_root) / model_name / "evaluation_predictions.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Evaluation artifact not found: {artifact_path}")

    data = torch.load(artifact_path, map_location="cpu", weights_only=False)
    y_true = data["y_true"]
    raw_predictions = data["y_pred"]
    task_type = data.get("task_type", "classification")

    if task_type == "classification":
        if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:
            y_probs = torch.softmax(raw_predictions, dim=1)
            y_pred = torch.argmax(raw_predictions, dim=1)
        else:
            y_probs = torch.sigmoid(raw_predictions.reshape(-1))
            y_pred = (y_probs >= 0.5).long()
    else:
        y_probs = None
        y_pred = raw_predictions

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "inputs": data.get("inputs"),
        "task_type": task_type,
        "num_classes": data.get("num_classes"),
    }


# =====================================================================
# 1. Exploratory Data Analysis (EDA)
# =====================================================================


def plot_correlation_heatmap(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: Optional[str] = None,
    top_k: Optional[int] = None,
    method: str = "pearson",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plots a feature correlation heatmap using Pearson, Spearman, or Kendall correlation.

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical features.
        target_col (Optional[str]): Target feature to filter top correlations.
        top_k (Optional[int]): Total number of top correlated features to display with target.
        method (str): Correlation method: 'pearson', 'spearman', or 'kendall'. Default is 'pearson'.
        figsize (Tuple[int, int]): Size of the matplotlib figure.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The heatmap figure is saved and displayed in place.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)

    if target_col and top_k and target_col in corr.columns:
        target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
        k_pos = top_k // 2
        k_neg = top_k - k_pos

        top_features = list(target_corr.head(k_pos).index) + list(target_corr.tail(k_neg).index)
        selected_cols = top_features + [target_col]
        corr = corr.loc[selected_cols, selected_cols]
        title = f"{method.capitalize()} Correlation Heatmap (Top {top_k} Features for '{target_col}')"
    else:
        title = f"Full Feature {method.capitalize()} Correlation Heatmap"

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, pad=15)

    fig.tight_layout()
    _save_figure(fig, dataset_name, "correlation_heatmap.png")


def plot_distribution(
    df: pd.DataFrame,
    dataset_name: str,
    columns: Optional[List[str]] = None,
    max_cols: int = 3,
    figsize_per_row: Tuple[int, int] = (15, 4),
) -> None:
    """
    Plots histograms with KDE curves for numerical features in a grid layout.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        columns (Optional[List[str]]): List of numeric columns to plot. If None, all numeric
            columns are plotted.
        max_cols (int): Maximum number of columns in the grid layout.
        figsize_per_row (Tuple[int, int]): Figure size per row of subplots.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The distribution figure is saved and displayed in place.
    """
    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)

    n_features = len(columns)
    if n_features == 0:
        raise ValueError("No numeric columns available to plot distribution.")

    rows = math.ceil(n_features / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, dataset_name, "distribution.png")


def plot_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plots a matrix identifying missing values patterns (black = missing, light = present).

    Args:
        df (pd.DataFrame): Input DataFrame to analyze for missing values.
        figsize (Tuple[int, int]): Figure size for the missing values matrix.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The missing values figure is saved and displayed in place.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False, cmap="binary", yticklabels=False, ax=ax)
    ax.set_title("Missing Values Matrix (Black = Null)", fontsize=14, pad=15)
    ax.set_ylabel("Samples (Rows)", fontsize=12)
    ax.set_xlabel("Features (Columns)", fontsize=12)

    fig.tight_layout()
    _save_figure(fig, dataset_name, "missing_values.png")


def plot_grouped_boxplots(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    num_cols: List[str],
    max_cols: int = 3,
    figsize_per_row: Tuple[int, int] = (15, 4),
) -> None:
    """
    Plots a grid of boxplots for numeric features grouped by target class categories.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        target_col (str): Target column name to group by.
        num_cols (List[str]): List of numeric columns to plot.
        max_cols (int): Maximum number of columns in the grid layout.
        figsize_per_row (Tuple[int, int]): Figure size per row of subplots.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The grouped boxplots figure is saved and displayed in place.
    """
    n_features = len(num_cols)
    rows = math.ceil(n_features / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette="Set2", hue=target_col, legend=False)
        axes[i].set_title(f"{col} by {target_col}", fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, dataset_name, "grouped_boxplots.png")


def plot_variable_types(
    df: pd.DataFrame,
    dataset_name: str,
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """
    Plots a summary bar chart showing counts of dataset feature data types.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        figsize (Tuple[int, int]): Figure size for the bar chart.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The variable types figure is saved and displayed in place.
    """
    dtypes_summary = df.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=dtypes_summary.index, y=dtypes_summary.values, ax=ax, palette="mako", hue=dtypes_summary.index, legend=False)

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    ax.set_title("Dataset Variable Types Breakdown", fontsize=14, pad=15)
    ax.set_xlabel("Data Type", fontsize=12)

    ax.set_ylabel("Number of Columns", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, dataset_name, "variable_types.png")


# =====================================================================
# 2. Dimensionality Reduction (PCA / t-SNE)
# =====================================================================


def plot_pca_variance(
    pca_info: Dict[str, Any],
    dataset_name: str,
    cut_off: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Plots 1x2 figure showing individual explained variance bars and cumulative variance line.

    Args:
        pca_info (Dict[str, Any]): Dictionary containing PCA results with keys:
            - "explained_variance_ratio": np.ndarray of explained variance ratios per component.
            - "cumulative_variance_ratio": np.ndarray of cumulative variance ratios.
        cut_off (Optional[float]): Optional threshold for cumulative variance to indicate elbow point.
        figsize (Tuple[int, int]): Figure size for the plots.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The PCA variance figure is saved and displayed in place.
    """
    explained_var = pca_info["explained_variance_ratio"]
    cum_var = pca_info["cumulative_variance_ratio"]
    components = np.arange(1, len(explained_var) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(components, explained_var, color="skyblue", edgecolor="navy", alpha=0.7)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax1.set_title("Individual Variance per Component", fontsize=14)

    ax2.plot(components, cum_var, marker="o", linestyle="-", color="crimson", linewidth=2)
    ax2.set_xlabel("Number of Components", fontsize=12)
    ax2.set_ylabel("Cumulative Variance Ratio", fontsize=12)
    ax2.set_title("Cumulative Variance Ratio", fontsize=14)
    ax2.set_ylim([0.0, 1.05])

    if cut_off is not None:
        ax2.axhline(y=cut_off, color="gray", linestyle="--", label=f"Cut-off ({cut_off:.0%})")
        idx = np.where(cum_var >= cut_off)[0]
        if len(idx) > 0:
            elbow_x = components[idx[0]]
            ax2.axvline(x=elbow_x, color="gray", linestyle=":")
            ax2.scatter(elbow_x, cum_var[idx[0]], color="black", zorder=5)
            ax2.annotate(f"{elbow_x} components", xy=(elbow_x, cum_var[idx[0]]),
                         xytext=(10, -20), textcoords="offset points", fontweight="bold")
        ax2.legend()

    fig.tight_layout()
    _save_figure(fig, dataset_name, "pca_variance.png")


def plot_embeddings_2d(
    embeddings: Union[np.ndarray, torch.Tensor],
    dataset_name: str,
    labels: Optional[Union[np.ndarray, list]] = None,
    title: str = "2D Embeddings Visualization",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plots 2D scatter plot for dimensional reduction embeddings.

    Args:
        embeddings (Union[np.ndarray, torch.Tensor]): 2D embeddings to plot.
        labels (Optional[Union[np.ndarray, list]]): Optional labels for coloring points.
        title (str): Title for the plot.
        figsize (Tuple[int, int]): Figure size for the scatter plot.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The 2D embeddings figure is saved and displayed in place.
    """
    emb = _to_numpy(embeddings)
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=_to_numpy(labels), cmap="tab10", alpha=0.8, s=35)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Target Class / Value")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], alpha=0.8, s=35, color="teal")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)

    fig.tight_layout()
    _save_figure(fig, dataset_name, "embeddings_2d.png")


def plot_embeddings_3d(
    embeddings: Union[np.ndarray, torch.Tensor],
    dataset_name: str,
    labels: Optional[Union[np.ndarray, list]] = None,
    title: str = "3D Embeddings Visualization",
    figsize: Tuple[int, int] = (9, 7),
) -> None:
    """
    Plots 3D scatter plot for dimensional reduction embeddings.

    Args:
        embeddings (Union[np.ndarray, torch.Tensor]): 3D embeddings to plot.
        labels (Optional[Union[np.ndarray, list]]): Optional labels for coloring points.
        title (str): Title for the plot.
        figsize (Tuple[int, int]): Figure size for the scatter plot.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The 3D embeddings figure is saved and displayed in place.
    """
    emb = _to_numpy(embeddings)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if labels is not None:
        scatter = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=_to_numpy(labels), cmap="tab10", alpha=0.8, s=30)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Target Class / Value")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], alpha=0.8, s=30, color="teal")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    fig.tight_layout()
    _save_figure(fig, dataset_name, "embeddings_3d.png")


# =====================================================================
# 3. Model Training & Evaluation
# =====================================================================


def plot_training_history(
    model_name: str,
    metric_name: str = "accuracy",
    figsize: Tuple[int, int] = (12, 5),
    checkpoint_root: Union[str, Path] = "checkpoints",
) -> None:
    """
    Plots 1x2 training history curves (Loss and Metric) comparing train vs validation.

    Args:
        model_name (str): Name of the model whose history is loaded.
        metric_name (str): Name of the metric to plot (e.g., "accuracy",
            "precision", "recall").
        figsize (Tuple[int, int]): Figure size for the plots.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.

    Returns:
        None: The training history figure is saved and displayed in place.
    """
    history_path = Path(checkpoint_root) / model_name / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as history_file:
        history = json.load(history_file)

    epochs = range(1, len(history.get("train_loss", history.get("loss", []))) + 1)
    train_loss = history.get("train_loss", history.get("loss", []))
    val_loss = history.get("val_loss", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, train_loss, label="Train Loss", color="dodgerblue", linewidth=2)
    if len(val_loss) > 0:
        ax1.plot(epochs, val_loss, label="Val Loss", color="coral", linewidth=2)
    ax1.set_title("Loss Trajectory", fontsize=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    train_metric = history.get(f"train_{metric_name}", history.get(metric_name, []))
    val_metric = history.get(f"val_{metric_name}", [])

    if len(train_metric) > 0:
        ax2.plot(epochs, train_metric, label=f"Train {metric_name.capitalize()}", color="dodgerblue", linewidth=2)
        if len(val_metric) > 0:
            ax2.plot(epochs, val_metric, label=f"Val {metric_name.capitalize()}", color="coral", linewidth=2)
        ax2.set_title(f"{metric_name.capitalize()} Trajectory", fontsize=14)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel(metric_name.capitalize())
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save_figure(fig, model_name, "training_history.png")


def plot_confusion_matrix(
    model_name: str,
    checkpoint_root: Union[str, Path] = "checkpoints",
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plots normalized or raw confusion matrix heatmap for classification evaluation.

    Args:
        model_name (str): Name of the model whose evaluation artifacts are loaded.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.
        class_names (Optional[List[str]]): List of class names for labeling axes. If None,
            integer labels are used.
        normalize (bool): Whether to normalize the confusion matrix values to proportions.
        figsize (Tuple[int, int]): Figure size for the heatmap.

    Returns:
        None: The confusion matrix figure is saved and displayed in place.
    """
    artifacts = load_evaluation_artifacts(model_name, checkpoint_root=checkpoint_root)
    y_true = artifacts["y_true"]
    y_pred = artifacts["y_pred"]

    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    cm = confusion_matrix(yt, yp)
    fmt = ".2f" if normalize else "d"
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax,
    )
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14, pad=15)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)

    fig.tight_layout()
    filename = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    _save_figure(fig, model_name, filename)


def plot_sample_predictions(
    model_name: str,
    class_names: Optional[List[str]] = None,
    max_samples: int = 15,
    figsize: Tuple[int, int] = (15, 9),
    checkpoint_root: Union[str, Path] = "checkpoints",
) -> None:
    """
    Plots a grid of sample images with titles colored green (correct) or red (incorrect).

    Args:
        model_name (str): Name of the model whose evaluation artifacts are loaded.
        class_names (Optional[List[str]]): List of class names for labeling. If None,
            integer labels are used.
        max_samples (int): Maximum number of samples to display in the grid.
        figsize (Tuple[int, int]): Figure size for the grid of images.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.

    Returns:
        None: The sample predictions figure is saved and displayed in place.
    """
    artifacts = load_evaluation_artifacts(model_name, checkpoint_root=checkpoint_root)
    if artifacts["inputs"] is None:
        raise ValueError("Evaluation artifacts do not contain inputs for sample predictions.")

    imgs = _to_numpy(artifacts["inputs"])
    yt = _to_numpy(artifacts["y_true"][: len(imgs)])
    yp = _to_numpy(artifacts["y_pred"][: len(imgs)])

    n_samples = min(len(imgs), max_samples)
    cols = 5
    rows = math.ceil(n_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_samples):
        img = imgs[i]
        # Transpose CHW to HWC if needed
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax = axes[i]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        true_lbl = class_names[int(yt[i])] if class_names else str(yt[i])
        pred_lbl = class_names[int(yp[i])] if class_names else str(yp[i])
        is_correct = int(yt[i]) == int(yp[i])

        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color="green" if is_correct else "red", fontsize=10)
        ax.axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, model_name, "sample_predictions.png")


def plot_roc_curves(
    model_name: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    checkpoint_root: Union[str, Path] = "checkpoints",
) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) curves and calculates AUC for binary or multi-class.

    Args:
        model_name (str): Name of the model whose evaluation artifacts are loaded.
        class_names (Optional[List[str]]): List of class names for labeling. If None,
            integer labels are used.
        figsize (Tuple[int, int]): Figure size for the ROC curves.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.

    Returns:
        None: The ROC curves figure is saved and displayed in place.
    """
    artifacts = load_evaluation_artifacts(model_name, checkpoint_root=checkpoint_root)
    if artifacts["y_probs"] is None:
        raise ValueError("ROC curves require classification evaluation artifacts.")

    yt = _to_numpy(artifacts["y_true"])
    yp = _to_numpy(artifacts["y_probs"])

    fig, ax = plt.subplots(figsize=figsize)

    if yp.ndim == 1 or yp.shape[1] == 1:
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="darkorange", lw=2)
    else:
        n_classes = yp.shape[1]
        for i in range(n_classes):
            binary_yt = (yt == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_yt, yp[:, i])
            roc_auc = auc(fpr, tpr)
            lbl = class_names[i] if class_names else f"Class {i}"
            ax.plot(fpr, tpr, lw=2, label=f"{lbl} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save_figure(fig, model_name, "roc_curves.png")


def plot_regression_residuals(
    model_name: str,
    figsize: Tuple[int, int] = (14, 5),
    checkpoint_root: Union[str, Path] = "checkpoints",
) -> None:
    """
    Plots 1x2 figure showing Predicted vs Actual scatter and Residuals vs Predicted plot.

    Args:
        model_name (str): Name of the model whose evaluation artifacts are loaded.
        figsize (Tuple[int, int]): Figure size for the plots.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoints.

    Returns:
        None: The regression residuals figure is saved and displayed in place.
    """
    artifacts = load_evaluation_artifacts(model_name, checkpoint_root=checkpoint_root)
    if artifacts["task_type"] != "regression":
        raise ValueError("Regression residuals require regression evaluation artifacts.")

    yt = _to_numpy(artifacts["y_true"]).flatten()
    yp = _to_numpy(artifacts["y_pred"]).flatten()
    residuals = yt - yp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(yp, yt, alpha=0.6, color="steelblue", edgecolor="k")
    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit")
    ax1.set_xlabel("Predicted Values", fontsize=12)
    ax1.set_ylabel("Actual Values", fontsize=12)
    ax1.set_title("Actual vs. Predicted", fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.scatter(yp, residuals, alpha=0.6, color="crimson", edgecolor="k")
    ax2.axhline(y=0, color="black", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Values", fontsize=12)
    ax2.set_ylabel("Residuals (Actual - Pred)", fontsize=12)
    ax2.set_title("Residual Analysis", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save_figure(fig, model_name, "regression_residuals.png")


def plot_model_comparison(
    models_input: Union[List[Dict[str, Any]], List[Union[str, Path]]],
    model_name: str,
    metric_name: str = "Accuracy",
    metrics: Optional[Union[List[float], Dict[str, float]]] = None,
    metric_key: str = "test_accuracy",
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Plots 1x2 figure comparing Model Metric vs Parameters and GFLOPs with K/M/B scales.

    Args:
        models_input (Union[List[Dict], List[str/Path]]): Either a list of dictionaries with model specs
            or a list of directory paths containing 'metadata.json'.
        metric_name (str): Display name for the metric axis (e.g., 'Accuracy', 'F1-Score').
        metrics (Optional[Union[List[float], Dict[str, float]]]): Explicit metric values passed as a
            list matching model_dirs order or a dict mapping model names/folder names to values.
        metric_key (str): Key to look for inside 'eval_results.json' if loading automatically from folder.
        figsize (Tuple[int, int]): Matplotlib figure dimensions.
        model_name (str): Name used as the output directory for the comparison figure.

    Returns:
        None: The model comparison figure is saved and displayed in place.
    """
    models_data = []

    # Check if input is a list of directory paths
    if models_input and isinstance(models_input[0], (str, Path)):
        for idx, model_dir in enumerate(models_input):
            model_path = Path(model_dir)
            meta_path = model_path / "metadata.json"
            eval_path = model_path / "eval_results.json"

            if not meta_path.exists():
                raise FileNotFoundError(f"Arquivo metadata.json não encontrado em: {model_path}")

            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            model_name = data.get("name", model_path.name)
            m_val = None

            # 1. Tenta obter métrica do parâmetro explícito 'metrics'
            if isinstance(metrics, dict):
                m_val = metrics.get(model_name, metrics.get(model_path.name))
            elif isinstance(metrics, list) and idx < len(metrics):
                m_val = metrics[idx]

            # 2. Se não foi passada explicitamente, busca no 'eval_results.json'
            if m_val is None and eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                    m_val = eval_data.get(metric_key)

            if m_val is None:
                raise ValueError(
                    f"Métrica para o modelo '{model_name}' não foi encontrada. "
                    f"Passe o parâmetro 'metrics' ou garanta que '{eval_path}' contenha a chave '{metric_key}'."
                )

            data["metric"] = m_val
            models_data.append(data)
    else:
        # Se já for a lista de dicionários
        models_data = models_input

    names = [d["name"] for d in models_data]
    metric_vals = [d["metric"] for d in models_data]
    params = [d.get("params", 0) for d in models_data]
    gflops = [d.get("gflops", 0) for d in models_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    formatter = ticker.FuncFormatter(_format_scale)

    # Subplot 1: Métricas vs Parâmetros
    ax1.scatter(params, metric_vals, color="darkorange", s=100, alpha=0.8, edgecolor="k")
    for i, name in enumerate(names):
        ax1.annotate(name, (params[i], metric_vals[i]), xytext=(5, 5), textcoords="offset points")
    ax1.set_xlabel("Number of Parameters", fontsize=12)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.set_title(f"{metric_name} vs. Model Parameters", fontsize=14)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Métricas vs GFLOPs
    ax2.scatter(gflops, metric_vals, color="mediumseagreen", s=100, alpha=0.8, edgecolor="k")
    for i, name in enumerate(names):
        ax2.annotate(name, (gflops[i], metric_vals[i]), xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("GFLOPs", fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f"{metric_name} vs. GFLOPs", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save_figure(fig, model_name, "model_comparison.png")


# =====================================================================
# 4. Explainable AI (XAI)
# =====================================================================


def plot_feature_maps(
    feature_maps: Union[torch.Tensor, np.ndarray],
    model_name: str,
    max_maps: int = 16,
    cols: int = 4,
    figsize_per_row: Tuple[int, int] = (12, 3),
) -> None:
    """
    Plots intermediate CNN activation maps in a grid layout.

    Args:
        feature_maps (Union[torch.Tensor, np.ndarray]): Feature maps to visualize.
        max_maps (int): Maximum number of feature maps to display.
        cols (int): Number of columns in the grid layout.
        figsize_per_row (Tuple[int, int]): Figure size per row of subplots.
        model_name (str): Model name used as the output directory.

    Returns:
        None: The feature maps figure is saved and displayed in place.
    """
    fmaps = _to_numpy(feature_maps)
    if fmaps.ndim == 4:
        fmaps = fmaps[0]  # Take first sample in batch

    n_channels = min(fmaps.shape[0], max_maps)
    rows = math.ceil(n_channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_channels):
        axes[i].imshow(fmaps[i], cmap="viridis")
        axes[i].set_title(f"Channel {i}", fontsize=10)
        axes[i].axis("off")

    for j in range(n_channels, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, model_name, "feature_maps.png")


def plot_grad_cam(
    image: Union[torch.Tensor, np.ndarray],
    model_name: str,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    title: str = "Grad-CAM Heatmap Overlay",
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Plots side-by-side comparison of original image and overlaid Grad-CAM heatmap.

    Args:
        image (Union[torch.Tensor, np.ndarray]): Original input image.
        heatmap (np.ndarray): Grad-CAM heatmap to overlay.
        alpha (float): Transparency level for heatmap overlay.
        title (str): Title for the heatmap overlay subplot.
        figsize (Tuple[int, int]): Figure size for the side-by-side plots.
        model_name (str): Model name used as the output directory.

    Returns:
        None: The Grad-CAM figure is saved and displayed in place.
    """
    img = _to_numpy(image)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax1.set_title("Original Input Image", fontsize=12)
    ax1.axis("off")

    ax2.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax2.imshow(heatmap, cmap="jet", alpha=alpha)
    ax2.set_title(title, fontsize=12)
    ax2.axis("off")

    fig.tight_layout()
    _save_figure(fig, model_name, "grad_cam.png")


def plot_feature_importance(
    importance_scores: np.ndarray,
    model_name: str,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    title: str = "Feature Importance Attributions",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plots horizontal bar chart ranking feature attributions/importance scores.

    Args:
        importance_scores (np.ndarray): Array of feature importance scores.
        feature_names (Optional[List[str]]): List of feature names corresponding to scores.
        top_n (Optional[int]): Number of top features to display. If None, all features
            are displayed.
        title (str): Title for the feature importance plot.
        figsize (Tuple[int, int]): Figure size for the bar chart.
        model_name (str): Model name used as the output directory.

    Returns:
        None: The feature importance figure is saved and displayed in place.
    """
    scores = _to_numpy(importance_scores).flatten()
    n_features = len(scores)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    df_imp = pd.DataFrame({"feature": feature_names, "importance": scores})
    df_imp["abs_importance"] = df_imp["importance"].abs()
    df_imp = df_imp.sort_values(by="abs_importance", ascending=True)

    if top_n and top_n < n_features:
        df_imp = df_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["crimson" if val < 0 else "teal" for val in df_imp["importance"]]

    ax.barh(df_imp["feature"], df_imp["importance"], color=colors, alpha=0.85)
    ax.axvline(x=0, color="black", linestyle="--", lw=1)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Attribution Value / Importance Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save_figure(fig, model_name, "feature_importance.png")
