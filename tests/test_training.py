"""Tests for training components."""

import pytest
import torch

from adaptive_quantization_aware_serving_with_dynamic_precision.models.model import create_model
from adaptive_quantization_aware_serving_with_dynamic_precision.training.trainer import Trainer
from adaptive_quantization_aware_serving_with_dynamic_precision.data.loader import get_dataloaders


def test_trainer_initialization(sample_config, device):
    """Test trainer initialization."""
    model = create_model(sample_config, num_classes=10)
    trainer = Trainer(model, sample_config, device)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.criterion is not None


def test_trainer_train_epoch(sample_config, device):
    """Test single epoch training."""
    model = create_model(sample_config, num_classes=10)
    trainer = Trainer(model, sample_config, device)

    train_loader, _, _ = get_dataloaders(sample_config)

    # Train for one epoch (with reduced data)
    metrics = trainer.train_epoch(train_loader, epoch=1)

    assert 'train_loss' in metrics
    assert 'train_acc' in metrics
    assert metrics['train_loss'] > 0
    assert 0 <= metrics['train_acc'] <= 100


def test_trainer_validate(sample_config, device):
    """Test validation."""
    model = create_model(sample_config, num_classes=10)
    trainer = Trainer(model, sample_config, device)

    _, val_loader, _ = get_dataloaders(sample_config)

    metrics = trainer.validate(val_loader)

    assert 'val_loss' in metrics
    assert 'val_acc' in metrics
    assert metrics['val_loss'] > 0
    assert 0 <= metrics['val_acc'] <= 100


def test_trainer_checkpoint_save_load(sample_config, device, tmp_path):
    """Test checkpoint saving and loading."""
    # Update checkpoint dir to tmp_path
    sample_config['logging']['checkpoint_dir'] = str(tmp_path / 'checkpoints')

    model = create_model(sample_config, num_classes=10)
    trainer = Trainer(model, sample_config, device)

    # Save checkpoint
    trainer.save_checkpoint(score=0.5, is_best=True)

    checkpoint_path = tmp_path / 'checkpoints' / 'best_model.pth'
    assert checkpoint_path.exists()

    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))
    assert trainer.current_epoch >= 0


def test_optimizer_creation(sample_config, device):
    """Test different optimizer types."""
    model = create_model(sample_config, num_classes=10)

    # Test AdamW
    sample_config['optimizer']['type'] = 'adamw'
    trainer = Trainer(model, sample_config, device)
    assert trainer.optimizer is not None

    # Test Adam
    sample_config['optimizer']['type'] = 'adam'
    trainer = Trainer(model, sample_config, device)
    assert trainer.optimizer is not None

    # Test SGD
    sample_config['optimizer']['type'] = 'sgd'
    trainer = Trainer(model, sample_config, device)
    assert trainer.optimizer is not None


def test_scheduler_creation(sample_config, device):
    """Test different scheduler types."""
    model = create_model(sample_config, num_classes=10)

    # Test cosine
    sample_config['lr_scheduler']['type'] = 'cosine'
    trainer = Trainer(model, sample_config, device)
    assert trainer.scheduler is not None

    # Test step
    sample_config['lr_scheduler']['type'] = 'step'
    sample_config['lr_scheduler']['step_size'] = 10
    sample_config['lr_scheduler']['gamma'] = 0.1
    trainer = Trainer(model, sample_config, device)
    assert trainer.scheduler is not None


def test_gradient_clipping(sample_config, device):
    """Test gradient clipping."""
    model = create_model(sample_config, num_classes=10)
    sample_config['training']['gradient_clip_norm'] = 1.0

    trainer = Trainer(model, sample_config, device)

    # Create dummy gradients
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param) * 10

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip_norm)

    # Check that gradients are clipped
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    assert total_norm <= trainer.grad_clip_norm * 1.1  # Small tolerance for numerical errors
