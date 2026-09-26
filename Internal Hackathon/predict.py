"""
predict.py - Standalone inference script for classifying any uploaded image.
Outputs top-k predictions with confidence percentages and visualizes results.
"""

import os
import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD


def load_inference_model(model_path: str, device: torch.device):
    """Loads trained checkpoint and class names."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")
    num_classes = checkpoint.get("num_classes", len(checkpoint.get("class_names", [])))
    class_names = checkpoint.get("class_names", [])

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(
    image_path: str,
    model_path: str = "output/best_model.pth",
    top_k: int = 3,
    save_plot_path: str = None
) -> Dict[str, Any]:
    """
    Runs classification inference on an input image.
    
    Args:
        image_path: Path to the uploaded image file
        model_path: Path to the trained .pth checkpoint
        top_k: Number of top class probabilities to return
        save_plot_path: Optional path to save visual prediction figure
        
    Returns:
        dict containing 'predicted_class', 'confidence', and 'top_k_predictions'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_inference_model(model_path, device)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load and preprocess image
    raw_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, top_k)

    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    top_predictions: List[Dict[str, Any]] = []
    for prob, idx in zip(top_probs, top_indices):
        label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        top_predictions.append({
            "class": label,
            "confidence": float(prob),
            "percentage": f"{prob * 100:.2f}%"
        })

    result = {
        "image_path": image_path,
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "percentage": top_predictions[0]["percentage"],
        "top_predictions": top_predictions
    }

    if save_plot_path:
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

        # Show input image
        ax_img.imshow(raw_img)
        ax_img.set_title(f"Prediction: {result['predicted_class']}\nConfidence: {result['percentage']}")
        ax_img.axis("off")

        # Show probability bar chart
        labels = [p["class"] for p in reversed(top_predictions)]
        confs = [p["confidence"] * 100 for p in reversed(top_predictions)]
        bars = ax_bar.barh(labels, confs, color="#4CAF50")
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.set_title("Top Predictions")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center")

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=200)
        plt.close()
        result["plot_path"] = save_plot_path
        print(f"Prediction plot saved to: {save_plot_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class of an uploaded image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="output/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=3, help="Top K predictions to output")
    parser.add_argument("--plot", type=str, default=None, help="Save prediction visualization image to path")

    args = parser.parse_args()
    res = predict_image(
        image_path=args.image,
        model_path=args.model,
        top_k=args.top_k,
        save_plot_path=args.plot
    )

    print("\n" + "=" * 45)
    print("           PREDICTION RESULT")
    print("=" * 45)
    print(f"Top Prediction : {res['predicted_class']} ({res['percentage']})")
    print("-" * 45)
    print("Rank | Class                  | Confidence")
    print("-" * 45)
    for rank, p in enumerate(res["top_predictions"], 1):
        print(f"{rank:<4} | {p['class']:<22} | {p['percentage']}")
    print("=" * 45)
