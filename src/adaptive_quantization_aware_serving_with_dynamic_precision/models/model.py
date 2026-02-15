"""Core model implementation with adaptive quantization support."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from .components import AccuracyPredictorNetwork

logger = logging.getLogger(__name__)


class AdaptiveQuantizedModel(nn.Module):
    """Main model with adaptive quantization support.

    Wraps a base classification model with an accuracy predictor network
    that determines optimal quantization precision at runtime.

    Args:
        base_model: Pre-trained or initialized base model.
        num_classes: Number of output classes.
        predictor_config: Configuration for accuracy predictor network.
        enable_predictor: Whether to use the accuracy predictor.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        predictor_config: Optional[Dict] = None,
        enable_predictor: bool = True
    ):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes
        self.enable_predictor = enable_predictor

        # Extract feature dimension from base model
        self.feature_dim = self._get_feature_dim()

        # Initialize accuracy predictor
        if enable_predictor and predictor_config is not None:
            self.accuracy_predictor = AccuracyPredictorNetwork(
                input_dim=self.feature_dim,
                hidden_dims=predictor_config.get('hidden_dims', [128, 64, 32]),
                num_precisions=3,
                dropout=predictor_config.get('dropout', 0.2),
                activation=predictor_config.get('activation', 'relu')
            )
        else:
            self.accuracy_predictor = None

        logger.info(
            f"Initialized AdaptiveQuantizedModel with {num_classes} classes, "
            f"feature_dim={self.feature_dim}, predictor_enabled={enable_predictor}"
        )

    def _get_feature_dim(self) -> int:
        """Extract feature dimension from base model."""
        if hasattr(self.base_model, 'fc'):
            # ResNet-style models
            return self.base_model.fc.in_features
        elif hasattr(self.base_model, 'classifier'):
            # VGG/AlexNet-style models
            if isinstance(self.base_model.classifier, nn.Sequential):
                return self.base_model.classifier[-1].in_features
            else:
                return self.base_model.classifier.in_features
        else:
            # Default fallback
            return 512

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            return_features: Whether to return intermediate features.

        Returns:
            Tuple of:
                - Classification outputs (batch_size, num_classes)
                - Features for predictor (batch_size, feature_dim) if return_features=True
                - Predictor scores (batch_size, num_precisions) if predictor enabled
        """
        # Extract features before final classification layer
        if hasattr(self.base_model, 'fc'):
            # ResNet-style: forward through all layers except fc
            features = self._forward_resnet_features(x)
        elif hasattr(self.base_model, 'classifier'):
            # VGG/AlexNet-style
            features = self._forward_vgg_features(x)
        else:
            # Generic fallback
            features = self.base_model(x)

        # Classification output
        if hasattr(self.base_model, 'fc'):
            outputs = self.base_model.fc(features)
        elif hasattr(self.base_model, 'classifier'):
            outputs = self.base_model.classifier(features)
        else:
            outputs = features

        # Accuracy predictor output
        predictor_scores = None
        if self.accuracy_predictor is not None and self.enable_predictor:
            predictor_scores = self.accuracy_predictor(features.detach())

        if return_features:
            return outputs, features, predictor_scores
        else:
            return outputs, None, predictor_scores

    def _forward_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through ResNet up to penultimate layer."""
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_vgg_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through VGG-style model up to penultimate layer."""
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def predict_precision(
        self,
        x: torch.Tensor,
        latency_budget: Optional[float] = None
    ) -> torch.Tensor:
        """Predict optimal precision for input samples.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            latency_budget: Optional latency constraint in milliseconds.

        Returns:
            Precision indices of shape (batch_size,).
        """
        if self.accuracy_predictor is None:
            # Default to INT8 if no predictor
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            _, features, _ = self.forward(x, return_features=True)
            precision_indices = self.accuracy_predictor.predict_precision(
                features, latency_budget
            )

        return precision_indices


def create_model(config: Dict, num_classes: int) -> AdaptiveQuantizedModel:
    """Factory function to create model from configuration.

    Args:
        config: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        Initialized AdaptiveQuantizedModel.

    Raises:
        ValueError: If model name is not supported.
    """
    model_name = config['model']['name'].lower()
    pretrained = config['model'].get('pretrained', True)

    # Create base model
    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        base_model = models.resnet34(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=pretrained)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        base_model = models.mobilenet_v2(pretrained=pretrained)
        base_model.classifier[-1] = nn.Linear(
            base_model.classifier[-1].in_features, num_classes
        )
    elif model_name == 'efficientnet_b0':
        base_model = models.efficientnet_b0(pretrained=pretrained)
        base_model.classifier[-1] = nn.Linear(
            base_model.classifier[-1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    logger.info(f"Created base model: {model_name}")

    # Check if accuracy predictor is enabled
    predictor_config = config.get('accuracy_predictor', {})
    enable_predictor = predictor_config.get('enabled', True)

    # Wrap in adaptive model
    model = AdaptiveQuantizedModel(
        base_model=base_model,
        num_classes=num_classes,
        predictor_config=predictor_config,
        enable_predictor=enable_predictor
    )

    return model
