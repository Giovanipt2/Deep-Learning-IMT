"""
Dimensionality Reduction Utilities.

Provides mathematical engines for Principal Component Analysis (PCA) and t-Distributed
Stochastic Neighbor Embedding (t-SNE), supporting automated feature reduction pipelines,
flexible thresholding, and seamless PyTorch/NumPy interoperation.
"""

import logging
from typing import Any, Optional, Union
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts PyTorch tensors or arbitrary sequences into a 2D NumPy float array.

    Args:
        x (Union[torch.Tensor, np.ndarray]): Input data, either a PyTorch tensor or a NumPy array.

    Returns:
        np.ndarray: 2D NumPy array of type float32.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x.astype(np.float32)


def fit_pca(
    x: Union[torch.Tensor, np.ndarray],
    n_components_or_cumulative_ratio: Optional[Union[int, float]] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fits Principal Component Analysis (PCA) on input data.

    Args:
        x (Union[torch.Tensor, np.ndarray]): Input data matrix of shape (N, D).
        n_components_or_cumulative_ratio (Optional[Union[int, float]]): Number of components to keep (int, e.g., 3)
            or ratio of cumulative variance to explain (float between 0.0 and 1.0, e.g., 0.95).
            If None, keeps all principal components.
        random_state (int): Seed for reproducibility. Defaults to 42.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'embeddings': Transformed dataset array of shape (N, K).
            - 'explained_variance_ratio': Array with variance explained per component.
            - 'cumulative_variance_ratio': Cumulative sum of explained variance ratios.
            - 'pca_model': Fitted sklearn PCA instance.
    """
    x_arr = _to_numpy(x)
    n_samples, n_features = x_arr.shape

    # Validate n_components_or_cumulative_ratio bounds
    if isinstance(n_components_or_cumulative_ratio, float) and not (0.0 < n_components_or_cumulative_ratio <= 1.0):
        raise ValueError("n_components_or_cumulative_ratio as float must be in range (0.0, 1.0].")
    if isinstance(n_components_or_cumulative_ratio, int) and n_components_or_cumulative_ratio > min(n_samples, n_features):
        logger.warning(
            f"Requested n_components_or_cumulative_ratio ({n_components_or_cumulative_ratio}) > min(n_samples, n_features) ({min(n_samples, n_features)}). "
            f"Capping n_components_or_cumulative_ratio to {min(n_samples, n_features)}."
        )
        n_components_or_cumulative_ratio = min(n_samples, n_features)

    pca = PCA(n_components_or_cumulative_ratio=n_components_or_cumulative_ratio, random_state=random_state)
    embeddings = pca.fit_transform(x_arr)
    explained_var = pca.explained_variance_ratio_

    return {
        "embeddings": embeddings,
        "explained_variance_ratio": explained_var,
        "cumulative_variance_ratio": np.cumsum(explained_var),
        "pca_model": pca,
    }


def fit_tsne(
    x: Union[torch.Tensor, np.ndarray],
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: Union[str, float] = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    pca_pre_reduction: Optional[Union[float, int, bool]] = None,
    auto_pca_threshold: int = 50,
    auto_pca_variance: float = 0.95,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Fits t-Distributed Stochastic Neighbor Embedding (t-SNE) with intelligent PCA pre-reduction.

    Args:
        x (Union[torch.Tensor, np.ndarray]): Input data matrix of shape (N, D).
        n_components (int): Dimension of the embedded space (typically 2 or 3). Defaults to 2.
        perplexity (float): Perplexity parameter related to number of nearest neighbors. Defaults to 30.0.
        learning_rate (Union[str, float]): Learning rate for optimization. Defaults to 'auto'.
        n_iter (int): Maximum number of iterations for optimization. Defaults to 1000.
        random_state (int): Seed for reproducibility. Defaults to 42.
        pca_pre_reduction (Optional[Union[float, int, bool]]):
            - If float (0 < v <= 1.0): Applies PCA keeping 'v' fraction of cumulative variance.
            - If int (v > 1): Applies PCA keeping 'v' principal components.
            - If True: Explicitly applies PCA with `auto_pca_variance`.
            - If False: Explicitly disables PCA pre-reduction.
            - If None: Automatically triggers PCA if number of features D > `auto_pca_threshold`.
        auto_pca_threshold (int): Feature count threshold to auto-trigger PCA when `pca_pre_reduction` is None. Defaults to 50.
        auto_pca_variance (float): Default variance ratio retained during auto PCA pre-reduction. Defaults to 0.95.
        **kwargs: Additional keyword arguments passed directly to sklearn.manifold.TSNE.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'embeddings': Transformed dataset array of shape (N, n_components).
            - 'pca_applied': Boolean indicating whether PCA pre-reduction was executed.
            - 'pca_info': Results dictionary from `fit_pca` if PCA was applied, else None.
            - 'tsne_model': Fitted sklearn TSNE instance.
    """
    x_arr = _to_numpy(x)
    n_samples, n_features = x_arr.shape

    should_run_pca = False
    pca_param: Optional[Union[int, float]] = None

    # Evaluate decision logic for PCA pre-reduction
    if pca_pre_reduction is False:
        should_run_pca = False
    elif pca_pre_reduction is True:
        should_run_pca = True
        pca_param = auto_pca_variance
    elif isinstance(pca_pre_reduction, (int, float)):
        should_run_pca = True
        pca_param = pca_pre_reduction
    elif pca_pre_reduction is None:
        if n_features > auto_pca_threshold:
            should_run_pca = True
            pca_param = auto_pca_variance
            logger.info(
                f"Feature dimension ({n_features}) exceeds auto threshold ({auto_pca_threshold}). "
                f"Automatically applying PCA pre-reduction to retain {auto_pca_variance:.0%} cumulative variance before t-SNE."
            )

    pca_info = None
    data_for_tsne = x_arr

    if should_run_pca and pca_param is not None:
        pca_info = fit_pca(x_arr, n_components_or_cumulative_ratio=pca_param, random_state=random_state)
        data_for_tsne = pca_info["embeddings"]
        logger.info(
            f"PCA pre-reduction completed: Reduced dimensions from {n_features} "
            f"to {data_for_tsne.shape[1]} components prior to t-SNE execution."
        )

    # Validate perplexity against sample count
    if perplexity >= n_samples:
        adjusted_perplexity = max(1.0, float(n_samples - 1) / 3.0)
        logger.warning(
            f"Perplexity ({perplexity}) must be strictly less than n_samples ({n_samples}). "
            f"Adjusting perplexity to {adjusted_perplexity:.2f}."
        )
        perplexity = adjusted_perplexity

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state,
        **kwargs,
    )

    embeddings = tsne.fit_transform(data_for_tsne)

    return {
        "embeddings": embeddings,
        "pca_applied": should_run_pca,
        "pca_info": pca_info,
        "tsne_model": tsne,
    }
