"""
Deep Learning Utilities Package.

Provides reusable infrastructure components for hardware acceleration, reproducibility,
checkpoint management, early stopping, structured logging, metric tracking, visualization,
dimensionality reduction, model explainability, and training execution.
"""

from .checkpoint import CheckpointManager
from .device import get_device, set_seed
from .dim_reduction import fit_pca, fit_tsne
from .early_stopping import EarlyStopping
from .explainability import (
    FeatureMapExtractor,
    GradCAM,
    compute_feature_attributions,
    compute_global_feature_importance,
)
from .logger import setup_logger
from .metrics import (
    MetricTracker,
    calculate_accuracy,
    calculate_cohen_kappa,
    calculate_mae,
    calculate_mse,
    calculate_precision_recall_f1,
    calculate_r2_score,
    calculate_rmse,
    calculate_roc_auc,
)
from utils.model_utils import get_model_gflops, get_model_parameters
from .trainer import Trainer
from .visualization import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_embeddings_2d,
    plot_embeddings_3d,
    plot_feature_importance,
    plot_feature_maps,
    plot_grad_cam,
    plot_grouped_boxplots,
    plot_image_samples,
    plot_missing_values,
    plot_model_comparison,
    plot_pca_variance,
    plot_regression_residuals,
    plot_roc_curves,
    plot_sample_predictions,
    plot_tabular_samples,
    plot_target_distribution,
    plot_training_history,
    plot_variable_types,
)

__all__ = [
    # Device & Setup
    "get_device",
    "set_seed",
    "setup_logger",
    # Training & Checkpoints
    "Trainer",
    "CheckpointManager",
    "EarlyStopping",
    # Metrics
    "MetricTracker",
    "calculate_accuracy",
    "calculate_precision_recall_f1",
    "calculate_roc_auc",
    "calculate_cohen_kappa",
    "calculate_mse",
    "calculate_rmse",
    "calculate_mae",
    "calculate_r2_score",
    # Dimensionality Reduction
    "fit_pca",
    "fit_tsne",
    # Explainability
    "FeatureMapExtractor",
    "GradCAM",
    "compute_feature_attributions",
    "compute_global_feature_importance",
    # Visualization Suite
    "plot_correlation_heatmap",
    "plot_distribution",
    "plot_missing_values",
    "plot_grouped_boxplots",
    "plot_variable_types",
    "plot_target_distribution",
    "plot_tabular_samples",
    "plot_image_samples",
    "plot_pca_variance",
    "plot_embeddings_2d",
    "plot_embeddings_3d",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_sample_predictions",
    "plot_roc_curves",
    "plot_regression_residuals",
    "plot_model_comparison",
    "plot_feature_maps",
    "plot_grad_cam",
    "plot_feature_importance",
    # Model Utilities
    "get_model_parameters",
    "get_model_gflops",
]
