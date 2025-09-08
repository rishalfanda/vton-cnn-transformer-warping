"""
Training Script untuk Segmentation Model
Bab IV 4.3.5 - Training Pipeline
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import create_dataloaders
from models.transunet import TransUNet, UNet
from losses.losses import CombinedSegmentationLoss
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_segmentation
from utils.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationTrainer:
    """
    Trainer class untuk model segmentasi
    Bab IV 4.3.5.1 - Segmentation Training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.build_model()
        
        # Initialize loss
        self.criterion = CombinedSegmentationLoss(
            dice_weight=config['segmentation']['dice_weight'],
            ce_weight=config['segmentation']['ce_weight'],
            boundary_weight=config['segmentation']['boundary_weight'],
            use_boundary=config['segmentation']['use_boundary_loss']
        )
        
        # Initialize optimizer
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Mixed precision training
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(
            num_classes=config['segmentation']['num_classes']
        )
        
        # Logging
        self.writer = SummaryWriter(self.config['training']['checkpoint_dir'] / 'logs')
        self.use_wandb = config.get('wandb', {}).get('enabled', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb', {}).get('project', 'vton-segmentation'),
                config=config
            )
        
        # Best model tracking
        self.best_miou = 0.0
        self.early_stop_counter = 0
    
    def setup_directories(self):
        """Create necessary directories"""
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self) -> nn.Module:
        """Build segmentation model based on config"""
        model_type = self.config['segmentation']['model_type']
        
        if model_type == 'transunet':
            model = TransUNet(
                img_size=tuple(self.config['preprocessing']['image_size']),
                in_channels=self.config['segmentation']['in_channels'],
                num_classes=self.config['segmentation']['num_classes'],
                encoder_channels=self.config['segmentation']['encoder_channels'],
                decoder_channels=self.config['segmentation']['decoder_channels'],
                hidden_size=self.config['segmentation']['hidden_size'],
                num_layers=self.config['segmentation']['num_layers'],
                num_heads=self.config['segmentation']['num_heads'],
                mlp_dim=self.config['segmentation']['mlp_dim'],
                dropout=self.config['segmentation']['dropout'],
                patch_size=self.config['segmentation']['patch_size'],
                use_checkpoint=self.config['training']['gradient_checkpointing']
            )
        elif model_type == 'unet':
            model = UNet(
                in_channels=self.config['segmentation']['in_channels'],
                num_classes=self.config['segmentation']['num_classes'],
                encoder_channels=self.config['segmentation']['encoder_channels'],
                decoder_channels=self.config['segmentation']['decoder_channels']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if self.config['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        logger.info(f"Built {model_type} model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        
        return model
    
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        opt_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        wd = self.config['training']['weight_decay']
        
        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        return optimizer
    
    def build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        sched_type = self.config['training']['scheduler']
        
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif sched_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, train_loader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            person_img = batch['person_img'].to(self.device)
            parse = batch['parse'].to(self.device) if 'parse' in batch else None
            
            if parse is None:
                logger.warning("No parse labels found, skipping batch")
                continue
            
            # Mixed precision training
            with autocast(enabled=self.use_amp):
                # Forward pass
                pred = self.model(person_img)
                
                # Calculate loss
                loss, loss_dict = self.criterion(pred, parse)
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.grad_accum_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.grad_accum_steps,
                'dice': loss_dict.get('dice_loss', 0),
                'boundary': loss_dict.get('boundary_loss', 0)
            })
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Train/{k}', v, global_step)
        
        avg_loss = epoch_loss / len(train_loader)
        
        return {'loss': avg_loss}
    
    def validate(self, val_loader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} - Validation")
            
            for batch in progress_bar:
                person_img = batch['person_img'].to(self.device)
                parse = batch['parse'].to(self.device) if 'parse' in batch else None
                
                if parse is None:
                    continue
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    pred = self.model(person_img)
                    loss, loss_dict = self.criterion(pred, parse)
                
                val_loss += loss.item()
                
                # Collect predictions for metrics
                pred_labels = pred.argmax(dim=1)
                all_preds.append(pred_labels.cpu())
                all_targets.append(parse.cpu())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics.compute(all_preds, all_targets)
        metrics['loss'] = val_loss / len(val_loader)
        
        # Log metrics
        logger.info(f"Validation - Epoch {epoch}: Loss={metrics['loss']:.4f}, mIoU={metrics['miou']:.4f}, PA={metrics['pixel_accuracy']:.4f}")
        
        # Tensorboard logging
        for k, v in metrics.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        
        # WandB logging
        if self.use_wandb:
            wandb.log({f'val_{k}': v for k, v in metrics.items()}, step=epoch)
        
        return metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['miou'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )
            
            # Save best model
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                self.early_stop_counter = 0
                
                best_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    best_path
                )
                logger.info(f"New best model saved with mIoU={self.best_miou:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.early_stop_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info(f"Training completed. Best mIoU: {self.best_miou:.4f}")
        
        # Clean up
        self.writer.close()
        if self.use_wandb:
            wandb.finish()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, default='configs/default.yml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create trainer
    trainer = SegmentationTrainer(config)
    
    # Resume if specified
    if args.resume:
        load_checkpoint(trainer.model, trainer.optimizer, args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()