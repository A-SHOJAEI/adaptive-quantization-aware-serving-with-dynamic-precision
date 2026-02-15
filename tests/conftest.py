"""Pytest configuration and fixtures."""

import pytest
import torch
import yaml
from pathlib import Path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'name': 'resnet18',
            'num_classes': 10,
            'pretrained': False
        },
        'accuracy_predictor': {
            'enabled': True,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2,
            'activation': 'relu'
        },
        'quantization': {
            'precisions': ['int8', 'fp16', 'fp32'],
            'qat_epochs': 2,
            'calibration_samples': 100
        },
        'training': {
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'gradient_clip_norm': 1.0,
            'mixed_precision': False
        },
        'lr_scheduler': {
            'type': 'cosine',
            'warmup_epochs': 1,
            'min_lr': 0.00001
        },
        'early_stopping': {
            'patience': 5,
            'min_delta': 0.0001
        },
        'data': {
            'dataset': 'cifar10',
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'num_workers': 0,
            'augmentation': False
        },
        'optimizer': {
            'type': 'adamw',
            'betas': [0.9, 0.999],
            'eps': 0.00000001
        },
        'serving': {
            'target_throughput': 2000,
            'max_accuracy_drop': 0.02,
            'latency_budget_ms': 15,
            'switch_overhead_ms': 0.5
        },
        'logging': {
            'log_interval': 10,
            'save_top_k': 2,
            'checkpoint_dir': 'checkpoints',
            'results_dir': 'results'
        },
        'tracking': {
            'use_mlflow': False,
            'use_wandb': False,
            'experiment_name': 'test-experiment'
        },
        'seed': 42
    }


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def sample_batch():
    """Sample batch of images and labels."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)
