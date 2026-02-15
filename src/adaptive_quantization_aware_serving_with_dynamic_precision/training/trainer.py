"""Training loop implementation with quantization-aware training."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import QuantizationAwareLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for quantization-aware model training.

    Implements training loop with:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpoint management

    Args:
        model: Model to train.
        config: Configuration dictionary.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training hyperparameters
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.grad_clip_norm = config['training'].get('gradient_clip_norm', 1.0)
        self.mixed_precision = config['training'].get('mixed_precision', True)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize loss function
        self.criterion = QuantizationAwareLoss(
            classification_weight=1.0,
            predictor_weight=0.1,
            quantization_weight=0.05,
            label_smoothing=0.1
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Early stopping
        self.early_stopping_patience = config['early_stopping'].get('patience', 10)
        self.early_stopping_delta = config['early_stopping'].get('min_delta', 0.0001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Checkpoint management
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = config['logging'].get('save_top_k', 3)
        self.checkpoint_scores = []

        # Tracking
        self.current_epoch = 0
        self.global_step = 0

        logger.info(f"Initialized Trainer with {self.epochs} epochs, lr={self.learning_rate}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_type = self.config['optimizer']['type'].lower()

        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.config['optimizer'].get('betas', [0.9, 0.999]),
                eps=self.config['optimizer'].get('eps', 1e-8)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"Created optimizer: {optimizer_type}")
        return optimizer

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler from configuration."""
        scheduler_type = self.config['lr_scheduler']['type'].lower()

        if scheduler_type == 'cosine':
            warmup_epochs = self.config['lr_scheduler'].get('warmup_epochs', 5)
            min_lr = self.config['lr_scheduler'].get('min_lr', 1e-5)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            step_size = self.config['lr_scheduler'].get('step_size', 30)
            gamma = self.config['lr_scheduler'].get('gamma', 0.1)

            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        logger.info(f"Created scheduler: {scheduler_type}")
        return scheduler

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs, features, predictor_scores = self.model(
                        inputs, return_features=True
                    )

                    # Create dummy predictor targets (will be updated in future iterations)
                    predictor_targets = None
                    if predictor_scores is not None:
                        # Target: prefer lower precision for easy samples
                        predictor_targets = torch.softmax(predictor_scores.detach(), dim=1)

                    loss_dict = self.criterion(
                        outputs, targets,
                        predictor_scores, predictor_targets,
                        model=self.model
                    )
                    loss = loss_dict['loss']

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs, features, predictor_scores = self.model(
                    inputs, return_features=True
                )

                predictor_targets = None
                if predictor_scores is not None:
                    predictor_targets = torch.softmax(predictor_scores.detach(), dim=1)

                loss_dict = self.criterion(
                    outputs, targets,
                    predictor_scores, predictor_targets,
                    model=self.model
                )
                loss = loss_dict['loss']

                loss.backward()

                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })

            self.global_step += 1

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        }

    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _, predictor_scores = self.model(inputs, return_features=True)

                predictor_targets = None
                if predictor_scores is not None:
                    predictor_targets = torch.softmax(predictor_scores.detach(), dim=1)

                loss_dict = self.criterion(
                    outputs, targets,
                    predictor_scores, predictor_targets,
                    model=self.model
                )
                loss = loss_dict['loss']

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total

        return {
            'val_loss': val_loss,
            'val_acc': val_acc
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlflow_client: Optional[any] = None
    ) -> Dict[str, any]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            mlflow_client: Optional MLflow client for logging.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting training...")
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            # MLflow logging
            if mlflow_client is not None:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        'train_loss': train_metrics['train_loss'],
                        'train_acc': train_metrics['train_acc'],
                        'val_loss': val_metrics['val_loss'],
                        'val_acc': val_metrics['val_acc'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()

            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(val_metrics['val_loss'], is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(val_metrics['val_loss'], is_best=False)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info("Training completed")
        return history

    def save_checkpoint(self, score: float, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            score: Validation score (lower is better).
            is_best: Whether this is the best checkpoint so far.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        best_path = self.checkpoint_dir / 'best_model.pth'

        if is_best:
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        # Save latest
        latest_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, latest_path)

        # Manage top-k checkpoints
        self.checkpoint_scores.append((score, latest_path))
        self.checkpoint_scores.sort(key=lambda x: x[0])

        if len(self.checkpoint_scores) > self.save_top_k:
            _, path_to_remove = self.checkpoint_scores.pop()
            if path_to_remove.exists() and path_to_remove != best_path:
                path_to_remove.unlink()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
