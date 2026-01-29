"""
Utility Functions
Image processing and transformation utilities
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess PIL Image for model input
    
    Args:
        image: PIL Image object
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert PIL to numpy array (RGB)
    img_np = np.array(image)
    
    # If image is RGBA, convert to RGB
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # If image is grayscale, convert to RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Resize
    img_resized = cv2.resize(img_np, target_size)
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    
    # Normalize using ImageNet statistics
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # Add batch dimension [1, C, H, W]
    img_batch = img_normalized.unsqueeze(0)
    
    return img_batch


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor back to displayable format
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    # Inverse normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    
    # Convert to numpy
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np


def postprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Postprocess model output mask
    
    Args:
        mask_tensor: Model output tensor [1, 1, H, W]
        threshold: Threshold for binary mask
        
    Returns:
        Binary mask as numpy array [H, W]
    """
    # Apply sigmoid
    mask = torch.sigmoid(mask_tensor)
    
    # Remove batch and channel dimensions
    mask = mask.squeeze().cpu().numpy()
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    return binary_mask


def create_overlay(
    original_image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay visualization
    
    Args:
        original_image: Original image [H, W, C]
        mask: Binary mask [H, W]
        color: RGB color for mask overlay
        alpha: Transparency (0=transparent, 1=opaque)
        
    Returns:
        Overlay image [H, W, C]
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Ensure original image is uint8
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1 else original_image.astype(np.uint8)
    
    # Resize mask to match original image size if needed
    if mask_binary.shape != original_image.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (original_image.shape[1], original_image.shape[0]))
    
    # Create colored mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask_binary > 0] = color
    
    # Blend
    overlay = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
    
    return overlay


def create_masked_image(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image (keep only segmented region)
    
    Args:
        original_image: Original image [H, W, C]
        mask: Binary mask [H, W]
        
    Returns:
        Masked image [H, W, C]
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Resize mask to match original image size if needed
    if mask_binary.shape != original_image.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (original_image.shape[1], original_image.shape[0]))
    
    # Apply mask
    mask_3channel = np.stack([mask_binary] * 3, axis=-1)
    masked_image = (original_image * (mask_3channel / 255.0)).astype(np.uint8)
    
    return masked_image


def resize_keeping_aspect_ratio(
    image: np.ndarray,
    max_size: int = 800
) -> np.ndarray:
    """
    Resize image keeping aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    return Image.fromarray(image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array"""
    return np.array(image)

