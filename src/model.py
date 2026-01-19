"""
CNN Model Architecture for AI Image Detection

Provides both a custom CNN and a transfer learning approach using ResNet18.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, DROPOUT_RATE


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for AI image detection.
    
    Architecture:
    - 4 convolutional blocks with batch normalization
    - Adaptive average pooling
    - 2 fully connected layers with dropout
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
        super(CustomCNN, self).__init__()
        
        # Convolutional blocks
        self.conv_block1 = self._make_conv_block(3, 32)
        self.conv_block2 = self._make_conv_block(32, 64)
        self.conv_block3 = self._make_conv_block(64, 128)
        self.conv_block4 = self._make_conv_block(128, 256)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with BatchNorm, ReLU, and MaxPool."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class ResNetDetector(nn.Module):
    """
    Transfer learning model using pretrained ResNet18.
    
    The pretrained weights are frozen by default, and only the
    classification head is trained.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetDetector, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class AIImageDetector(nn.Module):
    """
    Main AI Image Detector model.
    
    Wrapper that can use either CustomCNN or ResNet18 based on configuration.
    """
    
    def __init__(
        self,
        model_type: str = "resnet",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the AI Image Detector.
        
        Args:
            model_type: 'custom' for CustomCNN or 'resnet' for ResNet18
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights (ResNet only)
            freeze_backbone: Whether to freeze backbone weights (ResNet only)
        """
        super(AIImageDetector, self).__init__()
        
        self.model_type = model_type
        
        if model_type == "custom":
            self.model = CustomCNN(num_classes=num_classes)
        elif model_type == "resnet":
            self.model = ResNetDetector(
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        return predicted, confidence
    
    def get_num_parameters(self) -> dict:
        """Get the number of trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }


def create_model(
    model_type: str = "resnet",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> AIImageDetector:
    """
    Factory function to create an AI Image Detector model.
    
    Args:
        model_type: 'custom' for CustomCNN or 'resnet' for ResNet18
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ResNet only)
        freeze_backbone: Whether to freeze backbone weights (ResNet only)
        
    Returns:
        Initialized AIImageDetector model
    """
    return AIImageDetector(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    # Test the model architectures
    print("Testing model architectures...")
    
    # Test input
    x = torch.randn(1, 3, 224, 224)
    
    # Test Custom CNN
    custom_model = AIImageDetector(model_type="custom")
    custom_output = custom_model(x)
    print(f"Custom CNN output shape: {custom_output.shape}")
    print(f"Custom CNN parameters: {custom_model.get_num_parameters()}")
    
    # Test ResNet
    resnet_model = AIImageDetector(model_type="resnet")
    resnet_output = resnet_model(x)
    print(f"ResNet18 output shape: {resnet_output.shape}")
    print(f"ResNet18 parameters: {resnet_model.get_num_parameters()}")
    
    # Test prediction
    pred_class, confidence = resnet_model.predict(x)
    print(f"Prediction: class={pred_class.item()}, confidence={confidence.item():.4f}")
    
    print("Model architecture OK")
