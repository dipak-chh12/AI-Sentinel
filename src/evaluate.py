"""
Evaluation module for AI Image Detector

Provides comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrix, and ROC curves.
"""
import os
import sys
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES, TEST_DIR, MODEL_DIR
from src.model import create_model
from src.dataset import create_data_loaders


class Evaluator:
    """
    Evaluator class for computing and visualizing model performance metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str] = CLASS_NAMES,
        device: torch.device = DEVICE
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        
        self.predictions = []
        self.ground_truths = []
        self.probabilities = []
    
    def evaluate(self) -> dict:
        """
        Run evaluation on the test set.
        
        Returns:
            Dictionary containing all metrics
        """
        self.model.eval()
        self.predictions = []
        self.ground_truths = []
        self.probabilities = []
        
        print("Running evaluation...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                self.predictions.extend(predicted.cpu().numpy())
                self.ground_truths.extend(labels.numpy())
                self.probabilities.extend(probs[:, 1].cpu().numpy())  # Prob of AI-generated
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> dict:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, and confusion matrix
        """
        y_true = np.array(self.ground_truths)
        y_pred = np.array(self.predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                zero_division=0
            )
        }
        
        # Compute ROC-AUC if we have probabilities
        if len(self.probabilities) > 0:
            try:
                fpr, tpr, _ = roc_curve(y_true, self.probabilities)
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['fpr'] = fpr
                metrics['tpr'] = tpr
            except Exception as e:
                print(f"Could not compute ROC curve: {e}")
                metrics['roc_auc'] = None
        
        return metrics
    
    def print_report(self, metrics: dict):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
        
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        print(f"\nConfusion Matrix:")
        print(f"  {'':<15} {'Predicted':^25}")
        print(f"  {'':<15} {'Real':<12} {'AI-Generated':<12}")
        cm = metrics['confusion_matrix']
        if len(cm) == 2:
            print(f"  {'Actual Real':<15} {cm[0][0]:<12} {cm[0][1]:<12}")
            print(f"  {'Actual AI':<15} {cm[1][0]:<12} {cm[1][1]:<12}")
        print("="*60)
    
    def plot_confusion_matrix(
        self,
        metrics: dict,
        save_path: Optional[str] = None
    ):
        """
        Plot and optionally save the confusion matrix.
        
        Args:
            metrics: Dictionary containing confusion matrix
            save_path: Optional path to save the figure
        """
        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            title='Confusion Matrix',
            ylabel='True Label',
            xlabel='Predicted Label'
        )
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        metrics: dict,
        save_path: Optional[str] = None
    ):
        """
        Plot and optionally save the ROC curve.
        
        Args:
            metrics: Dictionary containing FPR, TPR, and AUC
            save_path: Optional path to save the figure
        """
        if 'fpr' not in metrics or 'tpr' not in metrics:
            print("ROC data not available")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(
            metrics['fpr'], metrics['tpr'],
            color='darkorange', lw=2,
            label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})'
        )
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.close()


def evaluate_model(
    model_path: str = BEST_MODEL_PATH,
    model_type: str = "resnet",
    save_plots: bool = True
) -> dict:
    """
    Main function to evaluate a trained model.
    
    Args:
        model_path: Path to the trained model weights
        model_type: Type of model ('custom' or 'resnet')
        save_plots: Whether to save confusion matrix and ROC plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model(model_type=model_type)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Using randomly initialized weights")
    
    # Load test data
    _, _, test_loader = create_data_loaders()
    
    if len(test_loader.dataset) == 0:
        print("\n" + "="*60)
        print("ERROR: No test data found!")
        print("Please add images to the following directories:")
        print("  - data/test/real/")
        print("  - data/test/ai_generated/")
        print("="*60 + "\n")
        return {}
    
    # Evaluate
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)
    
    # Save plots
    if save_plots and len(metrics.get('confusion_matrix', [])) > 0:
        plots_dir = os.path.join(MODEL_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            metrics,
            save_path=os.path.join(plots_dir, "confusion_matrix.png")
        )
        
        if metrics.get('roc_auc'):
            evaluator.plot_roc_curve(
                metrics,
                save_path=os.path.join(plots_dir, "roc_curve.png")
            )
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AI Image Detector")
    parser.add_argument("--model-path", type=str, default=BEST_MODEL_PATH,
                        help="Path to trained model weights")
    parser.add_argument("--model-type", type=str, default="resnet",
                        choices=["custom", "resnet"],
                        help="Model architecture")
    parser.add_argument("--no-plots", action="store_true",
                        help="Don't save evaluation plots")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        model_type=args.model_type,
        save_plots=not args.no_plots
    )
