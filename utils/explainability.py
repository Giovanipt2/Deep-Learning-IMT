"""
Model Explainability and Interpretability Utilities.

Provides components for extracting CNN feature maps, computing Grad-CAM
(Gradient-weighted Class Activation Maps) for convolutional architectures, and
computing Integrated Gradients feature attributions for arbitrary PyTorch models.
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
        torch.Tensor: Input converted to tensor on the specified device.
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
    target_class: Optional[int] = None,
    baseline: Optional[Union[torch.Tensor, np.ndarray]] = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Computes Integrated Gradients feature attributions for a given input sample.

    Args:
        model (nn.Module): PyTorch model.
        input_tensor (Union[torch.Tensor, np.ndarray]): Single input sample (1D or 2D tensor).
        target_class (Optional[int]): Target class index for classification. If None, uses max output.
        baseline (Optional[Union[torch.Tensor, np.ndarray]]): Baseline reference sample.
            Defaults to zeros vector matching input_tensor shape.
        steps (int): Number of Riemann interpolation steps. Defaults to 50.

    Returns:
        np.ndarray: Feature attribution scores array matching input_tensor shape.
    """
    model.eval()
    device = next(model.parameters()).device

    x = _to_tensor(input_tensor, device)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    if baseline is None:
        x0 = torch.zeros_like(x)
    else:
        x0 = _to_tensor(baseline, device)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

    # Generate interpolated paths between baseline (x0) and input (x)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    interpolated_inputs = torch.cat([x0 + alpha * (x - x0) for alpha in alphas], dim=0)
    interpolated_inputs.requires_grad = True

    model.zero_grad()
    outputs = model(interpolated_inputs)

    if outputs.ndim > 1:
        if target_class is None:
            target_class = int(torch.argmax(outputs[0]).item())
        scores = outputs[:, target_class]
    else:
        scores = outputs

    # Compute gradients along interpolated path
    grads = torch.autograd.grad(outputs=scores.sum(), inputs=interpolated_inputs)[0]

    # Riemann sum approximation of gradients integral
    avg_grads = torch.mean(grads[:-1], dim=0, keepdim=True)

    # Integrated Gradients = (x - x0) * average_gradients
    attributions = (x - x0) * avg_grads

    return attributions.squeeze(0).detach().cpu().numpy()
