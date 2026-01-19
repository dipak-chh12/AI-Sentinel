"""
Configuration settings for AI Image Detector
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data paths
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Image settings
IMAGE_SIZE = 224
NUM_CHANNELS = 3

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4

# Model settings
NUM_CLASSES = 2
CLASS_NAMES = ["real", "ai_generated"]
DROPOUT_RATE = 0.5

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model save path
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "ai_generated"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "ai_generated"), exist_ok=True)
