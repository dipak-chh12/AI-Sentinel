# AI Image Detector

A deep learning prototype system to distinguish AI-generated images from real images using CNN-based classification.

## 🚀 Features

- **CNN-based Classification**: Custom CNN and ResNet18 transfer learning options
- **Image Preprocessing**: Normalization, augmentation, and feature extraction
- **Training Pipeline**: Complete training loop with validation and early stopping
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix, ROC curves
- **Easy Inference**: Simple API for single image and batch prediction

## 📁 Project Structure

```
aiimagegenerator/
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   ├── train/
│   │   ├── real/         # Real images for training
│   │   └── ai_generated/ # AI-generated images for training
│   └── test/
│       ├── real/         # Real images for testing
│       └── ai_generated/ # AI-generated images for testing
├── models/               # Saved model weights
└── src/
    ├── preprocessing.py  # Image preprocessing pipeline
    ├── dataset.py        # Custom dataset loader
    ├── model.py          # CNN architectures
    ├── train.py          # Training script
    ├── evaluate.py       # Evaluation metrics
    └── predict.py        # Inference script
```

## 🛠️ Installation

1. **Clone/Navigate to the project directory**:
```bash
cd /Users/dipakchhetri/aiimagegenerator
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

Add your images to the appropriate directories:

- **Training Data**:
  - `data/train/real/` - Real photographs
  - `data/train/ai_generated/` - AI-generated images

- **Test Data**:
  - `data/test/real/` - Real photographs for testing
  - `data/test/ai_generated/` - AI-generated images for testing

**Recommended**: Start with at least 100 images per class for training.

### Dataset Sources

You can obtain datasets from:
- [Kaggle AI Art Detection datasets](https://www.kaggle.com/datasets)
- Generate AI images using DALL-E, Midjourney, or Stable Diffusion
- Collect real photos from Unsplash or similar platforms

## 🎯 Usage

### Training

Train the model with default settings (ResNet18):
```bash
python src/train.py
```

Train with custom CNN:
```bash
python src/train.py --model custom --epochs 30
```

Available training options:
```bash
python src/train.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | resnet | Model type: 'custom' or 'resnet' |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--demo` | False | Run in demo mode (2 epochs) |
| `--resume` | False | Resume from checkpoint |

### Evaluation

Evaluate the trained model:
```bash
python src/evaluate.py
```

This will output:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Saved plots in `models/plots/`

### Prediction

Predict a single image:
```bash
python src/predict.py --image path/to/image.jpg
```

Batch prediction on a directory:
```bash
python src/predict.py --image path/to/directory --batch
```

## 🧠 Model Architecture

### Custom CNN
```
Input (224×224×3)
    ↓
Conv2d(3→32) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d(64→128) + BatchNorm + ReLU + MaxPool
    ↓
Conv2d(128→256) + BatchNorm + ReLU + MaxPool
    ↓
AdaptiveAvgPool → Flatten → FC(256→128) → Dropout → FC(128→2)
    ↓
Output (Real / AI-Generated)
```

### ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Custom classification head for binary classification
- ~11M parameters (mostly frozen during initial training)

## 📈 Expected Performance

With sufficient training data (1000+ images per class), expect:
- **Accuracy**: 85-95%
- **F1-Score**: 0.85-0.95

Performance depends on:
- Dataset quality and diversity
- Balance between real and AI-generated images
- Types of AI generators used in training data

## 🔧 Configuration

Edit `config.py` to customize:
- Image size (default: 224×224)
- Batch size, learning rate, epochs
- Model save paths
- Data directories

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pretrained models and transforms
