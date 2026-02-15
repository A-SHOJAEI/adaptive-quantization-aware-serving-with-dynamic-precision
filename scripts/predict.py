#!/usr/bin/env python
"""Inference script for adaptive quantization-aware serving."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from adaptive_quantization_aware_serving_with_dynamic_precision.data.preprocessing import (
    get_transforms,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.models.model import create_model
from adaptive_quantization_aware_serving_with_dynamic_precision.utils.config import (
    get_device,
    load_config,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with adaptive quantization-aware model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or JSON file with image paths'
    )
    parser.add_argument(
        '--latency-budget',
        type=float,
        default=15.0,
        help='Latency budget in milliseconds for precision selection'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to return'
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Tuple of (model, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint")

    config = checkpoint['config']

    # Create model
    num_classes = config['model']['num_classes']
    model = create_model(config, num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded model from {checkpoint_path}")
    return model, config


def load_image(image_path: str, transform):
    """Load and preprocess a single image.

    Args:
        image_path: Path to image file.
        transform: Transform to apply to image.

    Returns:
        Preprocessed image tensor.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def predict_single(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    latency_budget: float,
    top_k: int = 5
):
    """Run inference on a single image.

    Args:
        model: Trained model.
        image_tensor: Preprocessed image tensor.
        device: Device to run inference on.
        latency_budget: Latency budget in milliseconds.
        top_k: Number of top predictions to return.

    Returns:
        Dictionary with predictions and metadata.
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        # Predict optimal precision
        if hasattr(model, 'predict_precision'):
            precision_idx = model.predict_precision(image_tensor, latency_budget=latency_budget)
            precision_map = {0: 'INT8', 1: 'FP16', 2: 'FP32'}
            selected_precision = precision_map[precision_idx.item()]
        else:
            selected_precision = 'FP32'

        # Run inference
        outputs, _, _ = model(image_tensor)

        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class_id': int(idx.item()),
                'confidence': float(prob.item())
            })

    return {
        'predictions': predictions,
        'selected_precision': selected_precision,
        'latency_budget_ms': latency_budget
    }


def main():
    """Main inference function."""
    args = parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train a model first using: python scripts/train.py")
        return

    # Load model
    logger.info("Loading model...")
    model, config = load_model_from_checkpoint(args.checkpoint)

    # Get device
    device = get_device()
    model = model.to(device)
    model.eval()

    # Get transforms
    dataset_name = config['data']['dataset']
    transforms_dict = get_transforms(dataset_name, augmentation=False)
    transform = transforms_dict['val']

    # Process input
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    all_predictions = []

    # Check if input is JSON or image
    if input_path.suffix == '.json':
        # Load list of images from JSON
        with open(input_path, 'r') as f:
            image_list = json.load(f)

        if not isinstance(image_list, list):
            image_list = [image_list]

        logger.info(f"Processing {len(image_list)} images...")

        for item in image_list:
            if isinstance(item, dict):
                image_path = item.get('path', item.get('image', ''))
            else:
                image_path = item

            image_tensor = load_image(image_path, transform)
            if image_tensor is None:
                continue

            result = predict_single(
                model, image_tensor, device, args.latency_budget, args.top_k
            )
            result['image_path'] = str(image_path)
            all_predictions.append(result)

    else:
        # Single image file
        logger.info(f"Processing single image: {input_path}")

        image_tensor = load_image(str(input_path), transform)
        if image_tensor is not None:
            result = predict_single(
                model, image_tensor, device, args.latency_budget, args.top_k
            )
            result['image_path'] = str(input_path)
            all_predictions.append(result)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    logger.info(f"Saved predictions to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)

    for i, pred in enumerate(all_predictions, 1):
        print(f"\nImage {i}: {pred['image_path']}")
        print(f"  Selected Precision: {pred['selected_precision']}")
        print(f"  Top Predictions:")
        for j, p in enumerate(pred['predictions'][:3], 1):
            print(f"    {j}. Class {p['class_id']}: {p['confidence']*100:.2f}%")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
