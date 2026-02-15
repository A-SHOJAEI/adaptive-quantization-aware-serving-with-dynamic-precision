"""Custom model components including loss functions and accuracy predictor."""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AccuracyPredictorNetwork(nn.Module):
    """Lightweight network to predict per-sample accuracy across precision levels.

    This network takes intermediate activations from the main model and predicts
    which quantization precision (INT8/FP16/FP32) would achieve target accuracy
    with minimal latency.

    Args:
        input_dim: Dimension of input features (typically from model penultimate layer).
        hidden_dims: List of hidden layer dimensions.
        num_precisions: Number of precision levels to predict for.
        dropout: Dropout probability.
        activation: Activation function name ('relu', 'gelu', 'leaky_relu').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_precisions: int = 3,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_precisions = num_precisions

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError(f"Unknown activation: {activation}")

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer: predict confidence scores for each precision level
        layers.append(nn.Linear(prev_dim, num_precisions))

        self.network = nn.Sequential(*layers)
        logger.info(f"Initialized AccuracyPredictorNetwork with {len(hidden_dims)} hidden layers")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict precision-level confidence scores.

        Args:
            features: Input features of shape (batch_size, input_dim).

        Returns:
            Precision scores of shape (batch_size, num_precisions).
            Higher score means higher predicted accuracy for that precision.
        """
        return self.network(features)

    def predict_precision(
        self,
        features: torch.Tensor,
        latency_budget: Optional[float] = None
    ) -> torch.Tensor:
        """Predict optimal precision level for each sample.

        Args:
            features: Input features of shape (batch_size, input_dim).
            latency_budget: Optional latency constraint in milliseconds.

        Returns:
            Precision indices of shape (batch_size,) where:
                0 = INT8 (fastest, lowest accuracy)
                1 = FP16 (medium speed/accuracy)
                2 = FP32 (slowest, highest accuracy)
        """
        scores = self.forward(features)

        if latency_budget is not None:
            # Apply latency-aware masking
            # Assume relative latencies: INT8=1x, FP16=1.5x, FP32=2.5x
            latency_multipliers = torch.tensor([1.0, 1.5, 2.5], device=scores.device)

            # Mask out precisions that exceed latency budget
            # This is a simplified heuristic
            mask = latency_multipliers <= (latency_budget / 10.0)
            scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

        # Select precision with highest confidence score
        return torch.argmax(scores, dim=1)


class QuantizationAwareLoss(nn.Module):
    """Custom loss function for quantization-aware training with accuracy predictor.

    Combines classification loss with accuracy predictor training and quantization
    regularization to optimize the model for multi-precision deployment.

    Args:
        classification_weight: Weight for classification loss.
        predictor_weight: Weight for accuracy predictor loss.
        quantization_weight: Weight for quantization regularization.
        label_smoothing: Label smoothing factor for classification.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        predictor_weight: float = 0.1,
        quantization_weight: float = 0.05,
        label_smoothing: float = 0.1
    ):
        super().__init__()

        self.classification_weight = classification_weight
        self.predictor_weight = predictor_weight
        self.quantization_weight = quantization_weight

        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.predictor_loss = nn.MSELoss()

        logger.info(
            f"Initialized QuantizationAwareLoss with weights: "
            f"cls={classification_weight}, pred={predictor_weight}, "
            f"quant={quantization_weight}"
        )

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        predictor_scores: Optional[torch.Tensor] = None,
        predictor_targets: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            outputs: Model classification outputs of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).
            predictor_scores: Accuracy predictor outputs of shape (batch_size, num_precisions).
            predictor_targets: Target precision scores of shape (batch_size, num_precisions).
            model: Optional model reference for quantization regularization.

        Returns:
            Dictionary containing total loss and individual loss components.
        """
        # Classification loss
        cls_loss = self.classification_loss(outputs, targets)
        total_loss = self.classification_weight * cls_loss

        losses = {
            'classification_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }

        # Accuracy predictor loss
        if predictor_scores is not None and predictor_targets is not None:
            pred_loss = self.predictor_loss(predictor_scores, predictor_targets)
            total_loss = total_loss + self.predictor_weight * pred_loss
            losses['predictor_loss'] = pred_loss.item()
            losses['total_loss'] = total_loss.item()

        # Quantization regularization: penalize large weight magnitudes
        # to make model more amenable to quantization
        if model is not None and self.quantization_weight > 0:
            quant_reg = 0.0
            num_params = 0

            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # L2 regularization on weights to encourage quantization-friendly values
                    quant_reg = quant_reg + torch.norm(param, p=2)
                    num_params += 1

            if num_params > 0:
                quant_reg = quant_reg / num_params
                total_loss = total_loss + self.quantization_weight * quant_reg
                losses['quantization_reg'] = quant_reg.item()
                losses['total_loss'] = total_loss.item()

        # Return total loss as tensor for backprop
        return {'loss': total_loss, 'losses': losses}


class PrecisionSwitchingPolicy(nn.Module):
    """Policy network for dynamic precision switching decisions.

    Makes real-time decisions about which precision to use based on:
    - Current system load
    - Input complexity
    - Latency budget
    - Accuracy requirements

    Args:
        feature_dim: Dimension of input features.
        num_precisions: Number of available precision levels.
    """

    def __init__(self, feature_dim: int, num_precisions: int = 3):
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),  # +2 for latency_budget and load_factor
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_precisions)
        )

    def forward(
        self,
        features: torch.Tensor,
        latency_budget: torch.Tensor,
        load_factor: torch.Tensor
    ) -> torch.Tensor:
        """Compute precision selection logits.

        Args:
            features: Input features of shape (batch_size, feature_dim).
            latency_budget: Latency budget in ms of shape (batch_size, 1).
            load_factor: Current system load (0-1) of shape (batch_size, 1).

        Returns:
            Logits for each precision level of shape (batch_size, num_precisions).
        """
        # Concatenate all inputs
        policy_input = torch.cat([features, latency_budget, load_factor], dim=1)
        return self.policy_net(policy_input)
