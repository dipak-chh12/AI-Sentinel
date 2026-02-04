import os
from typing import Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, CLASS_NAMES
from src.preprocessing import get_train_transforms, get_val_transforms


class AIImageDataset(Dataset):

    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        class_names: List[str] = None
    ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        
        # Build class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            print(f"Warning: No images found in {data_dir}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all image paths and their corresponding labels.
        
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            label = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    image_path = os.path.join(class_dir, filename)
                    samples.append((image_path, label))
        
        return samples
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), color="black")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of samples across classes.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {cls: 0 for cls in self.class_names}
        
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        
        return distribution


def create_data_loaders(
    train_dir: str = TRAIN_DIR,
    test_dir: str = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = AIImageDataset(
        data_dir=train_dir,
        transform=get_train_transforms()
    )
    
    test_dataset = AIImageDataset(
        data_dir=test_dir,
        transform=get_val_transforms()
    )
    
    # Split training data into train and validation
    total_train = len(train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    if total_train > 0:
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation dataset with val transforms
        val_dataset = AIImageDataset(
            data_dir=train_dir,
            transform=get_val_transforms()
        )
        val_indices = val_subset.indices
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    else:
        train_subset = train_dataset
        val_subset = train_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_sample_batch(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch from a data loader.
    
    Args:
        data_loader: PyTorch DataLoader
        
    Returns:
        Tuple of (images, labels) tensors
    """
    for images, labels in data_loader:
        return images, labels
    return None, None


if __name__ == "__main__":
    # Test the dataset and data loaders
    print("Testing dataset loading...")
    
    # Create a test dataset
    train_dataset = AIImageDataset(
        data_dir=TRAIN_DIR,
        transform=get_train_transforms()
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Class distribution: {train_dataset.get_class_distribution()}")
    
    if len(train_dataset) > 0:
        # Get a sample
        image, label = train_dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample label: {label} ({train_dataset.idx_to_class[label]})")
    
    print("Dataset loading OK")
