"""Tests for model components."""

import pytest
import torch

from adaptive_quantization_aware_serving_with_dynamic_precision.models.components import (
    AccuracyPredictorNetwork,
    QuantizationAwareLoss,
)
from adaptive_quantization_aware_serving_with_dynamic_precision.models.model import (
    AdaptiveQuantizedModel,
    create_model,
)


def test_accuracy_predictor_network():
    """Test accuracy predictor network."""
    predictor = AccuracyPredictorNetwork(
        input_dim=512,
        hidden_dims=[128, 64],
        num_precisions=3,
        dropout=0.2
    )

    # Test forward pass
    features = torch.randn(4, 512)
    scores = predictor(features)

    assert scores.shape == (4, 3)

    # Test precision prediction
    precision_indices = predictor.predict_precision(features)
    assert precision_indices.shape == (4,)
    assert torch.all((precision_indices >= 0) & (precision_indices < 3))


def test_accuracy_predictor_with_latency_budget():
    """Test accuracy predictor with latency budget."""
    predictor = AccuracyPredictorNetwork(
        input_dim=512,
        hidden_dims=[128, 64],
        num_precisions=3
    )

    features = torch.randn(4, 512)
    precision_indices = predictor.predict_precision(features, latency_budget=10.0)

    assert precision_indices.shape == (4,)


def test_quantization_aware_loss(sample_batch):
    """Test quantization aware loss."""
    loss_fn = QuantizationAwareLoss(
        classification_weight=1.0,
        predictor_weight=0.1,
        quantization_weight=0.05
    )

    images, labels = sample_batch
    outputs = torch.randn(4, 10)

    # Test with only classification loss
    result = loss_fn(outputs, labels)

    assert 'loss' in result
    assert 'losses' in result
    assert 'classification_loss' in result['losses']
    assert isinstance(result['loss'], torch.Tensor)


def test_quantization_aware_loss_with_predictor(sample_batch):
    """Test loss with predictor scores."""
    loss_fn = QuantizationAwareLoss()

    images, labels = sample_batch
    outputs = torch.randn(4, 10)
    predictor_scores = torch.randn(4, 3)
    predictor_targets = torch.softmax(torch.randn(4, 3), dim=1)

    result = loss_fn(
        outputs, labels,
        predictor_scores, predictor_targets
    )

    assert 'predictor_loss' in result['losses']


def test_create_model(sample_config):
    """Test model creation from config."""
    model = create_model(sample_config, num_classes=10)

    assert isinstance(model, AdaptiveQuantizedModel)
    assert model.num_classes == 10
    assert model.accuracy_predictor is not None


def test_create_model_without_predictor(sample_config):
    """Test model creation without predictor."""
    sample_config['accuracy_predictor']['enabled'] = False
    model = create_model(sample_config, num_classes=10)

    assert isinstance(model, AdaptiveQuantizedModel)
    assert model.accuracy_predictor is None


def test_adaptive_model_forward(sample_config, sample_batch):
    """Test forward pass of adaptive model."""
    model = create_model(sample_config, num_classes=10)
    model.eval()

    images, labels = sample_batch

    with torch.no_grad():
        outputs, features, predictor_scores = model(images, return_features=True)

    assert outputs.shape == (4, 10)
    assert features.shape[0] == 4
    assert predictor_scores.shape == (4, 3)


def test_adaptive_model_predict_precision(sample_config, sample_batch):
    """Test precision prediction."""
    model = create_model(sample_config, num_classes=10)
    model.eval()

    images, labels = sample_batch

    with torch.no_grad():
        precision_indices = model.predict_precision(images)

    assert precision_indices.shape == (4,)
    assert torch.all((precision_indices >= 0) & (precision_indices < 3))


def test_model_output_shapes(sample_config):
    """Test that model produces correct output shapes."""
    model = create_model(sample_config, num_classes=10)

    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)

    with torch.no_grad():
        outputs, _, _ = model(images)

    assert outputs.shape == (batch_size, 10)
