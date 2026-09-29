"""
Model Explainability and Interpretability Utilities.

Provides components for extracting CNN feature maps, computing Grad-CAM
(Gradient-weighted Class Activation Maps) for convolutional architectures, and
computing Integrated Gradients feature attributions for arbitrary PyTorch models
across both classification and regression tasks (local and global importance).
"""

import logging
from typing import Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _to_tensor(x: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Ensures input is a PyTorch tensor on the target device with floating point dtype.

    Args:
        x (Union[torch.Tensor, np.ndarray]): Input data.
        device (torch.device): Target device for tensor.

    Returns:
        torch.Tensor: Tensor on the specified device with dtype=torch.float32.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x.to(device, dtype=torch.float32)


class FeatureMapExtractor:
    """
    Extracts intermediate feature maps from a target module within a PyTorch neural network.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initializes the FeatureMapExtractor.

        Args:
            model (nn.Module): The PyTorch neural network model.
            target_layer (nn.Module): The internal layer from which to extract feature maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps: Optional[torch.Tensor] = None
        self._hook_handle = self.target_layer.register_forward_hook(self._hook_fn)


    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """
        Forward hook callback storing activation maps.

        Args:
            module (nn.Module): The module to which the hook is registered.
            input (Any): Input to the module.
            output (torch.Tensor): Output from the module (feature maps).
        """
        self.feature_maps = output.detach()


    def remove_hook(self) -> None:
        """Removes the forward hook from the target module."""
        if self._hook_handle is not None:
            self._hook_handle.remove()


    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input x through the model and returns captured feature maps.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Feature maps tensor of shape (B, C, H, W) or (B, C, L).
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)

        if self.feature_maps is None:
            raise RuntimeError("Failed to extract feature maps. Ensure target_layer is part of forward pass.")

        return self.feature_maps


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps for CNNs.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initializes GradCAM.

        Args:
            model (nn.Module): PyTorch convolutional neural network model.
            target_layer (nn.Module): Target convolutional layer (usually final Conv2d layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._forward_hook = self.target_layer.register_forward_hook(self._forward_hook_fn)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook_fn)


    def _forward_hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        """
        Forward hook callback storing activation maps.

        Args:
            module (nn.Module): The module to which the hook is registered.
            input (Any): Input to the module.
            output (torch.Tensor): Output from the module (feature maps).
        """
        self.activations = output


    def _backward_hook_fn(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """
        Backward hook callback storing gradients of the target layer's output.

        Args:
            module (nn.Module): The module to which the hook is registered.
            grad_input (Any): Gradients with respect to the input of the module.
            grad_output (Any): Gradients with respect to the output of the module.
        """
        self.gradients = grad_output[0]


    def remove_hooks(self) -> None:
        """Removes forward and backward hooks."""
        if self._forward_hook is not None:
            self._forward_hook.remove()
        if self._backward_hook is not None:
            self._backward_hook.remove()


    def generate_heatmap(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap for a single image sample.

        Args:
            input_image (torch.Tensor): Input image tensor of shape (1, C, H, W).
            target_class (Optional[int]): Target class index to explain. If None, uses predicted class.

        Returns:
            np.ndarray: Normalized 2D heatmap array of shape (H, W) with values in range [0, 1].
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)

        input_image = input_image.to(device)
        input_image.requires_grad = True

        self.model.zero_grad()
        output = self.model(input_image)

        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        score = output[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks failed to capture activations/gradients. Check target_layer.")

        # Pool gradients across spatial dimensions
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize CAM to match original image spatial dimensions
        img_h, img_w = input_image.shape[2], input_image.shape[3]
        cam = F.interpolate(cam, size=(img_h, img_w), mode="bilinear", align_corners=False)

        cam_arr = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_arr.min(), cam_arr.max()
        if cam_max > cam_min:
            cam_arr = (cam_arr - cam_min) / (cam_max - cam_min)
        else:
            cam_arr = np.zeros_like(cam_arr)

        return cam_arr


def compute_feature_attributions(
    model: nn.Module,
    input_tensor: Union[torch.Tensor, np.ndarray],
    task_type: str = "classification",
    target_class: Optional[int] = None,
    baseline: Optional[Union[torch.Tensor, np.ndarray]] = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Computes Integrated Gradients feature attributions for single samples or batches.

    Args:
        model (nn.Module): PyTorch model.
        input_tensor (Union[torch.Tensor, np.ndarray]): Input tensor of shape (D,) or (N, D).
        task_type (str): Task type, either 'classification' or 'regression'. Defaults to 'classification'.
        target_class (Optional[int]): Target class index for classification. If None, uses model's top prediction.
            Ignored if task_type='regression'.
        baseline (Optional[Union[torch.Tensor, np.ndarray]]): Reference baseline tensor.
            Defaults to zeros vector matching input_tensor shape.
        steps (int): Number of Riemann interpolation steps. Defaults to 50.

    Returns:
        np.ndarray: Feature attribution scores array matching input_tensor shape.
            Positive values indicate positive contribution to output; negative values indicate opposing impact.
    """
    if task_type not in ("classification", "regression"):
        raise ValueError("task_type must be either 'classification' or 'regression'.")

    model.eval()
    device = next(model.parameters()).device

    x = _to_tensor(input_tensor, device)
    is_single_sample = x.ndim == 1
    if is_single_sample:
        x = x.unsqueeze(0)  # Shape becomes (1, D)

    if baseline is None:
        x0 = torch.zeros_like(x)
    else:
        x0 = _to_tensor(baseline, device)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

    # Interpolate paths from baseline to inputs across steps
    # shape: (steps + 1, batch_size, D)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)

    # Batch computation of path steps
    attributions_list = []

    # Process sample by sample to avoid GPU memory overflow during step interpolation
    for i in range(x.shape[0]):
        sample_x = x[i : i + 1]  # (1, D)
        sample_x0 = x0[i : i + 1] if x0.shape[0] > 1 else x0  # (1, D)

        interpolated_inputs = torch.cat([sample_x0 + alpha * (sample_x - sample_x0) for alpha in alphas], dim=0)
        interpolated_inputs.requires_grad = True

        model.zero_grad()
        outputs = model(interpolated_inputs)

        if task_type == "classification":
            if outputs.ndim > 1 and outputs.shape[1] > 1:
                if target_class is None:
                    # Use top predicted class of the un-interpolated target sample
                    target_c = int(torch.argmax(outputs[-1]).item())
                else:
                    target_c = target_class
                scores = outputs[:, target_c]
            else:
                scores = outputs.squeeze()
        else:  # regression
            scores = outputs.squeeze()

        # Compute gradients along interpolation path
        grads = torch.autograd.grad(outputs=scores.sum(), inputs=interpolated_inputs)[0]

        # Average gradients across steps and multiply by delta (x - x0)
        avg_grads = torch.mean(grads[:-1], dim=0, keepdim=True)
        attr = (sample_x - sample_x0) * avg_grads
        attributions_list.append(attr.detach().cpu().numpy())

    attributions = np.vstack(attributions_list)

    if is_single_sample:
        return attributions.squeeze(0)
    return attributions


def compute_global_feature_importance(
    model: nn.Module,
    dataset_loader_or_tensor: Union[torch.Tensor, np.ndarray],
    task_type: str = "classification",
    target_class: Optional[int] = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Computes global feature importance across multiple dataset samples by averaging
    mean absolute Integrated Gradients attributions.

    Args:
        model (nn.Module): PyTorch model.
        dataset_loader_or_tensor (Union[torch.Tensor, np.ndarray]): Data batch or matrix of shape (N, D).
        task_type (str): Task mode ('classification' or 'regression').
        target_class (Optional[int]): Optional class target for classification analysis.
        steps (int): Interpolation steps. Defaults to 50.

    Returns:
        np.ndarray: 1D array of shape (D,) containing global importance scores for each feature.
    """
    local_attributions = compute_feature_attributions(
        model=model,
        input_tensor=dataset_loader_or_tensor,
        task_type=task_type,
        target_class=target_class,
        steps=steps,
    )

    # Mean absolute value across all dataset samples (axis 0)
    global_importance = np.mean(np.abs(local_attributions), axis=0)
    return global_importance
