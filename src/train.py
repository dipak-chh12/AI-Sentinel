"""
Training script for AI Image Detector

Handles training loop, validation, checkpointing, and learning rate scheduling.
"""
import os
import sys
import argparse
import time
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, BEST_MODEL_PATH, CHECKPOINT_PATH, CLASS_NAMES
)
from src.model import create_model
from src.dataset import create_data_loaders, AIImageDataset
from src.preprocessing import get_train_transforms, get_val_transforms


class Trainer:
    """
    Trainer class for AI Image Detector.
    
    Handles the complete training pipeline including:
    - Training loop
    - Validation
    - Model checkpointing
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: torch.device = DEVICE,
        checkpoint_path: str = CHECKPOINT_PATH,
        best_model_path: str = BEST_MODEL_PATH
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        
        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"  ✓ Saved best model with accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self) -> int:
        """
        Load a training checkpoint if it exists.
        
        Returns:
            Starting epoch number
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            self.history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")
            return start_epoch
        return 0
    
    def train(
        self,
        num_epochs: int = NUM_EPOCHS,
        early_stopping_patience: int = 10,
        resume: bool = False
    ) -> dict:
        """
        Run the complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop after N epochs without improvement
            resume: Whether to resume from checkpoint
            
        Returns:
            Training history dictionary
        """
        start_epoch = self.load_checkpoint() if resume else 0
        
        print(f"\n{'='*60}")
        print(f"Training AI Image Detector")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"{'='*60}")
        
        return self.history


def train_model(
    model_type: str = "resnet",
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    demo: bool = False
) -> dict:
    """
    Main function to train the AI Image Detector.
    
    Args:
        model_type: 'custom' or 'resnet'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        demo: If True, run in demo mode with minimal data
        
    Returns:
        Training history
    """
    print(f"Creating {model_type} model...")
    model = create_model(model_type=model_type)
    print(f"Model parameters: {model.get_num_parameters()}")
    
    print("Loading data...")
    train_loader, val_loader, _ = create_data_loaders(batch_size=batch_size)
    
    if len(train_loader.dataset) == 0:
        print("\n" + "="*60)
        print("ERROR: No training data found!")
        print("Please add images to the following directories:")
        print("  - data/train/real/")
        print("  - data/train/ai_generated/")
        print("="*60 + "\n")
        return {}
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate
    )
    
    history = trainer.train(num_epochs=epochs)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train AI Image Detector")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["custom", "resnet"],
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with minimal training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    if args.demo:
        args.epochs = min(args.epochs, 2)
        print("Running in DEMO mode with reduced epochs")
    
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        demo=args.demo
    )


if __name__ == "__main__":
    main()
