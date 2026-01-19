"""
Inference script for AI Image Detector

Provides prediction functionality for single images and batches.
"""
import os
import sys
import argparse
from typing import Tuple, List, Union
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES
from src.model import create_model
from src.preprocessing import load_and_preprocess_image, get_inference_transforms


class Predictor:
    """
    Predictor class for making inference on images.
    """
    
    def __init__(
        self,
        model_path: str = BEST_MODEL_PATH,
        model_type: str = "resnet",
        device: torch.device = DEVICE,
        class_names: List[str] = CLASS_NAMES
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model weights
            model_type: Type of model architecture
            device: Device to run inference on
            class_names: List of class names
        """
        self.device = device
        self.class_names = class_names
        self.transform = get_inference_transforms()
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = create_model(model_type=model_type)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file not found. Using random weights.")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def predict(self, image: Union[str, Image.Image]) -> dict:
        """
        Make prediction on a single image.
        
        Args:
            image: Either a file path or PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        if isinstance(image, str):
            if not os.path.exists(image):
                return {"error": f"Image not found: {image}"}
            image_tensor = load_and_preprocess_image(image, self.transform)
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
        else:
            return {"error": "Invalid image format"}
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_class = self.class_names[predicted_idx.item()]
        
        return {
            "prediction": predicted_class,
            "confidence": confidence.item(),
            "is_ai_generated": predicted_idx.item() == 1,
            "probabilities": {
                self.class_names[0]: probabilities[0][0].item(),
                self.class_names[1]: probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for path in image_paths:
            result = self.predict(path)
            result["image_path"] = path
            results.append(result)
        
        return results
    
    def print_prediction(self, result: dict):
        """Print a formatted prediction result."""
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "="*50)
        if "image_path" in result:
            print(f"Image: {result['image_path']}")
        print("="*50)
        
        prediction = result['prediction']
        confidence = result['confidence'] * 100
        
        # Visual indicator
        if result['is_ai_generated']:
            indicator = "🤖 AI-GENERATED"
            color_code = "\033[91m"  # Red
        else:
            indicator = "📷 REAL"
            color_code = "\033[92m"  # Green
        
        reset_code = "\033[0m"
        
        print(f"\n  Result: {color_code}{indicator}{reset_code}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"\n  Probability Distribution:")
        print(f"    Real:         {result['probabilities']['real']*100:.2f}%")
        print(f"    AI-Generated: {result['probabilities']['ai_generated']*100:.2f}%")
        print("="*50 + "\n")


def predict_image(
    image_path: str,
    model_path: str = BEST_MODEL_PATH,
    model_type: str = "resnet"
) -> dict:
    """
    Convenience function to predict a single image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to trained model weights
        model_type: Type of model architecture
        
    Returns:
        Prediction dictionary
    """
    predictor = Predictor(model_path=model_path, model_type=model_type)
    result = predictor.predict(image_path)
    predictor.print_prediction(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="AI Image Detector - Prediction")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to analyze")
    parser.add_argument("--model-path", type=str, default=BEST_MODEL_PATH,
                        help="Path to trained model weights")
    parser.add_argument("--model-type", type=str, default="resnet",
                        choices=["custom", "resnet"],
                        help="Model architecture")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple images (provide directory path)")
    
    args = parser.parse_args()
    
    if args.batch and os.path.isdir(args.image):
        # Batch prediction
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        
        predictor = Predictor(model_path=args.model_path, model_type=args.model_type)
        results = predictor.predict_batch(image_paths)
        
        print(f"\nProcessed {len(results)} images:")
        for result in results:
            predictor.print_prediction(result)
        
        # Summary
        ai_count = sum(1 for r in results if r.get('is_ai_generated', False))
        real_count = len(results) - ai_count
        print(f"\nSummary: {real_count} Real, {ai_count} AI-Generated")
    else:
        # Single image prediction
        predict_image(
            image_path=args.image,
            model_path=args.model_path,
            model_type=args.model_type
        )


if __name__ == "__main__":
    main()
