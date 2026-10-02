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
from torch.utils.data import DataLoader, Dataset


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
    if not output_name or output_name in {"dataset", "model"}:
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
    if df.empty:
        raise ValueError("Cannot plot correlations for an empty DataFrame.")
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'.")
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns available to plot correlations.")
    corr = num_df.corr(method=method)

    if target_col and target_col not in corr.columns:
        raise ValueError(f"Target column must be numeric for correlation analysis: {target_col}")

    if target_col and top_k:
        target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
        k_pos = top_k // 2
        k_neg = top_k - k_pos

        top_features = list(target_corr.head(k_pos).index) + list(target_corr.tail(k_neg).index)
        top_features = list(dict.fromkeys(top_features))
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
    target_col: Optional[str] = None,
) -> None:
    """
    Plots histograms with KDE curves for numerical features in a grid layout.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        columns (Optional[List[str]]): List of numeric columns to plot. If None, all numeric
            columns are plotted.
        max_cols (int): Maximum number of columns in the grid layout.
        figsize_per_row (Tuple[int, int]): Figure size per row of subplots.
        target_col (Optional[str]): Optional target column to append to the plot. Discrete
            targets are displayed as counts instead of KDE curves.
        dataset_name (str): Dataset name used as the output directory.

    Returns:
        None: The distribution figure is saved and displayed in place.
    """
    if df.empty:
        raise ValueError("Cannot plot distributions for an empty DataFrame.")
    if max_cols <= 0:
        raise ValueError("max_cols must be greater than zero.")
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    else:
        columns = list(columns)

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric and target_col not in non_numeric:
        raise TypeError(f"Distribution columns must be numeric: {non_numeric}")
    if target_col is not None and target_col not in columns:
        columns.append(target_col)

    n_features = len(columns)
    if n_features == 0:
        raise ValueError("No numeric columns available to plot distribution.")

    rows = math.ceil(n_features / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(columns):
        is_target = col == target_col
        values = df[col].dropna()
        is_discrete = is_target and (
            not pd.api.types.is_numeric_dtype(df[col]) or values.nunique() <= 20
        )
        use_kde = not is_discrete and values.nunique() > 1
        sns.histplot(
            df[col],
            kde=use_kde,
            discrete=is_discrete,
            ax=axes[i],
            color="darkorange" if is_target else "steelblue",
        )
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _save_figure(fig, dataset_name, "distribution.png")


def plot_target_distribution(
    target: Union[pd.Series, np.ndarray, torch.Tensor, List[Any]],
    dataset_name: str,
    target_name: str = "Target",
    max_categories: int = 20,
    figsize: Tuple[int, int] = (9, 6),
) -> None:
    """Plots class balance for discrete targets or a histogram for continuous targets."""
    if max_categories <= 0:
        raise ValueError("max_categories must be greater than zero.")

    values = _to_numpy(target).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot plot an empty target.")
    values = values[~pd.isna(values)]
    if values.size == 0:
        raise ValueError("Target contains no non-null values.")

    unique_values = pd.unique(values)
    is_discrete = not np.issubdtype(values.dtype, np.number) or len(unique_values) <= max_categories
    fig, ax = plt.subplots(figsize=figsize)

    if is_discrete:
        counts = pd.Series(values).value_counts().sort_index()
        percentages = counts / counts.sum() * 100
        bars = ax.bar(counts.index.astype(str), counts.values, color="teal")
        ax.set_ylabel("Count")
        ax.set_xlabel(target_name)
        ax.set_title(f"{target_name} Distribution ({len(counts)} categories)")
        for bar, percentage in zip(bars, percentages):
            ax.annotate(
                f"{percentage:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
        if len(counts) > 8:
            ax.tick_params(axis="x", rotation=45)
    else:
        sns.histplot(values, kde=values.size > 1 and np.ptp(values) > 0, ax=ax, color="teal")
        ax.set_xlabel(target_name)
        ax.set_ylabel("Count")
        ax.set_title(f"{target_name} Distribution")

    fig.tight_layout()
    _save_figure(fig, dataset_name, "target_distribution.png")


def plot_tabular_samples(
    df: pd.DataFrame,
    dataset_name: str,
    n_samples: int = 5,
    columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> None:
    """Displays randomly selected rows from a tabular dataset as a table."""
    if df.empty:
        raise ValueError("Cannot plot samples from an empty DataFrame.")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if columns is None:
        selected_columns = list(df.columns)
    else:
        selected_columns = list(columns)
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")
    if not selected_columns:
        raise ValueError("At least one column must be selected.")

    samples = df.sample(n=min(n_samples, len(df)), random_state=random_state)
    display_df = samples.loc[:, selected_columns].copy()
    display_df.index = display_df.index.map(str)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=display_df.astype(str).values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title(f"Random Tabular Samples ({len(display_df)} rows)", pad=20)

    fig.tight_layout()
    _save_figure(fig, dataset_name, "tabular_samples.png")


def plot_image_samples(
    data: Union[Dataset, DataLoader, torch.Tensor, np.ndarray, List[Any]],
    dataset_name: str,
    n_samples: int = 5,
    labels: Optional[Union[torch.Tensor, np.ndarray, List[Any]]] = None,
    random_state: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    cols: int = 5,
    figsize_per_row: Tuple[int, int] = (15, 3),
) -> None:
    """Displays random image samples with labels, shapes, and channel information."""
    if n_samples <= 0 or cols <= 0:
        raise ValueError("n_samples and cols must be greater than zero.")

    source = data.dataset if isinstance(data, DataLoader) else data
    if isinstance(source, (Dataset, torch.Tensor, np.ndarray, list, tuple)) and not hasattr(source, "__len__"):
        raise ValueError("Image data must provide a length.")
    n_available = len(source)
    if n_available == 0:
        raise ValueError("Cannot plot samples from an empty image dataset.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_available, size=min(n_samples, n_available), replace=False)
    label_values = None if labels is None else _to_numpy(labels).reshape(-1)
    if label_values is not None and len(label_values) < n_available:
        raise ValueError("labels must contain at least one value per image.")

    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows))
    axes = np.atleast_1d(axes).flatten()

    for plot_index, sample_index in enumerate(indices):
        raw_sample = source[int(sample_index)]
        sample_label = label_values[int(sample_index)] if label_values is not None else None
        if isinstance(raw_sample, (tuple, list)):
            image = raw_sample[0]
            if sample_label is None and len(raw_sample) > 1:
                sample_label = _to_numpy(raw_sample[1]).reshape(-1)[0]
        else:
            image = raw_sample

        image_array = _to_numpy(image)
        original_shape = tuple(image_array.shape)
        if image_array.ndim == 3 and image_array.shape[0] in (1, 3, 4):
            image_array = np.moveaxis(image_array, 0, -1)
        if image_array.ndim == 3 and image_array.shape[-1] == 1:
            image_array = image_array[..., 0]
        if image_array.ndim not in (2, 3):
            raise ValueError(f"Expected a 2D image or a 3D image with channels, got {original_shape}.")

        display_image = image_array.astype(float)
        if display_image.min() < 0 or display_image.max() > 1:
            image_min, image_max = display_image.min(), display_image.max()
            if image_max > image_min:
                display_image = (display_image - image_min) / (image_max - image_min)

        ax = axes[plot_index]
        ax.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
        channels = 1 if display_image.ndim == 2 else display_image.shape[-1]
        label_text = ""
        if sample_label is not None:
            label_index = int(sample_label) if np.issubdtype(np.asarray(sample_label).dtype, np.integer) else None
            label_text = class_names[label_index] if class_names and label_index is not None else str(sample_label)
        ax.set_title(f"idx={sample_index} | shape={original_shape} | C={channels}" + (f"\nlabel={label_text}" if label_text else ""), fontsize=9)
        ax.axis("off")

    for empty_index in range(len(indices), len(axes)):
        axes[empty_index].axis("off")

    fig.tight_layout()
    _save_figure(fig, dataset_name, "image_samples.png")


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
    if df.empty:
        raise ValueError("Cannot plot missing values for an empty DataFrame.")

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
    if df.empty:
        raise ValueError("Cannot plot grouped boxplots for an empty DataFrame.")
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    if not num_cols:
        raise ValueError("At least one numeric column must be provided.")
    if max_cols <= 0:
        raise ValueError("max_cols must be greater than zero.")

    missing_columns = [col for col in num_cols if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")
    non_numeric = [col for col in num_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"Boxplot columns must be numeric: {non_numeric}")

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
    model_names: List[str],
    dataset_name: str,
    metric_name: str = "Accuracy",
    metric_key: str = "test_accuracy",
    checkpoint_root: Union[str, Path] = "checkpoints",
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Plots 1x2 figure comparing Model Metric vs Parameters and GFLOPs with K/M/B scales.

    Args:
        model_names (List[str]): Names of the models to compare. Each name must
            correspond to a directory under checkpoint_root.
        dataset_name (str): Dataset name used as the output directory.
        metric_name (str): Display name for the metric axis (e.g., 'Accuracy', 'F1-Score').
        metric_key (str): Key to look for inside 'eval_results.json' if loading automatically from folder.
        checkpoint_root (Union[str, Path]): Root directory containing model checkpoint directories.
        figsize (Tuple[int, int]): Matplotlib figure dimensions.

    Returns:
        None: The model comparison figure is saved and displayed in place.
    """
    models_data = []

    for model_name in model_names:
        model_path = Path(checkpoint_root) / model_name
        meta_path = model_path / "metadata.json"
        eval_path = model_path / "eval_results.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {eval_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        metric_value = eval_data.get(metric_key)
        if metric_value is None:
            raise ValueError(f"Metric '{metric_key}' was not found in: {eval_path}")

        data["name"] = data.get("name", model_name)
        data["metric"] = metric_value
        models_data.append(data)

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
    _save_figure(fig, f"{dataset_name}", "model_comparison.png")


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
