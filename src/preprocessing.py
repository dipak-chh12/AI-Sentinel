"""
Image preprocessing, normalization, and data augmentation pipeline
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """
    Get training transforms with data augmentation.
    
    Includes:
    - Random resized crop
    - Random horizontal flip
    - Random rotation
    - Color jitter
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Get validation/test transforms without augmentation.
    
    Only includes:
    - Resize
    - Center crop
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms():
    """
    Get transforms for single image inference.
    Same as validation transforms.
    """
    return get_val_transforms()


def load_and_preprocess_image(image_path: str, transform=None) -> torch.Tensor:
    """
    Load an image from path and apply preprocessing.
    
    Args:
        image_path: Path to the image file
        transform: Optional transform to apply. If None, uses inference transforms.
        
    Returns:
        Preprocessed image tensor of shape (1, C, H, W)
    """
    if transform is None:
        transform = get_inference_transforms()
    
    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor image for visualization.
    
    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        Denormalized numpy array suitable for plotting
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Clone and move to CPU
    tensor = tensor.clone().cpu()
    
    # Denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to HWC format for plotting
    return tensor.permute(1, 2, 0).numpy()


class ImagePreprocessor:
    """
    Wrapper class for image preprocessing operations.
    """
    
    def __init__(self, mode: str = "inference"):
        """
        Initialize preprocessor with specified mode.
        
        Args:
            mode: One of 'train', 'val', or 'inference'
        """
        self.mode = mode
        if mode == "train":
            self.transform = get_train_transforms()
        elif mode == "val":
            self.transform = get_val_transforms()
        else:
            self.transform = get_inference_transforms()
    
    def __call__(self, image):
        """Apply preprocessing to an image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def process_batch(self, image_paths: list) -> torch.Tensor:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch tensor of shape (N, C, H, W)
        """
        tensors = []
        for path in image_paths:
            tensor = load_and_preprocess_image(path, self.transform)
            tensors.append(tensor)
        
        return torch.cat(tensors, dim=0)


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    print(f"Train transforms: {train_transform}")
    print(f"Validation transforms: {val_transform}")
    
    # Test with a dummy image
    dummy_image = Image.new("RGB", (300, 300), color="red")
    
    train_output = train_transform(dummy_image)
    val_output = val_transform(dummy_image)
    
    print(f"Train output shape: {train_output.shape}")
    print(f"Val output shape: {val_output.shape}")
    print("Preprocessing OK")
