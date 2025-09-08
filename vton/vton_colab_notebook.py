# %% [markdown]
# # VTON Pipeline - End-to-End Implementation
# ## Virtual Try-On dengan TransUNet dan Boundary-aware Loss
# 
# Implementasi sesuai Bab IV Tesis - Metodologi Penelitian
# 
# ### Features:
# - **Segmentasi**: TransUNet (Hybrid CNN-Transformer) dengan boundary-aware loss
# - **Warping**: TPS/Flow-based/Neural warping dengan feedback loop
# - **Dataset**: VITON-HD dengan support alias folder names
# - **Auto-config**: Deteksi GPU dan optimasi memory otomatis
# - **Eksperimen**: Baseline, Proposed, dan Ablation studies

# %% [markdown]
# ## 1. Setup Environment

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q wandb tensorboard matplotlib seaborn scikit-learn
!pip install -q opencv-python albumentations
!pip install -q lpips pytorch-fid

# %%
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
else:
    print("No GPU available, using CPU")

# %% [markdown]
# ## 2. Auto-Configuration Based on GPU

# %%
def detect_gpu_config():
    """Detect GPU and return optimal configuration"""
    if not torch.cuda.is_available():
        return {
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'amp_dtype': 'fp16',
            'gradient_checkpointing': True
        }
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # GPU configurations
    configs = {
        't4': {
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'amp_dtype': 'fp16',
            'gradient_checkpointing': True
        },
        'p100': {
            'batch_size': 6,
            'gradient_accumulation_steps': 1,
            'amp_dtype': 'fp16',
            'gradient_checkpointing': True
        },
        'v100': {
            'batch_size': 8,
            'gradient_accumulation_steps': 1,
            'amp_dtype': 'fp16',
            'gradient_checkpointing': False
        },
        'a100': {
            'batch_size': 16,
            'gradient_accumulation_steps': 1,
            'amp_dtype': 'bf16',
            'gradient_checkpointing': False
        }
    }
    
    # Match GPU
    for gpu_key, config in configs.items():
        if gpu_key in gpu_name:
            print(f"Detected {gpu_key.upper()}, using optimized config")
            return config
    
    # Default config
    print("Using default configuration")
    return configs['t4']

# Get auto config
auto_config = detect_gpu_config()
print(f"Configuration: {auto_config}")

# %% [markdown]
# ## 3. Setup Project Structure

# %%
# Create project directories
project_dirs = [
    '/content/vton',
    '/content/vton/data',
    '/content/vton/models',
    '/content/vton/losses',
    '/content/vton/utils',
    '/content/vton/preprocess',
    '/content/configs',
    '/content/cache',
    '/content/drive/MyDrive/vton_outputs',
    '/content/drive/MyDrive/vton_outputs/checkpoints',
    '/content/drive/MyDrive/vton_outputs/results'
]

for dir_path in project_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

print("Project structure created successfully!")

# %% [markdown]
# ## 4. Create Configuration Files

# %%
# Create main configuration
config = {
    'dataset': {
        'data_root': '/content',
        'subset': 'all',
        'train_size': 0.8,
        'val_size': 0.2,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 2
    },
    'dataset_aliases': {
        'image_train': ['dataset-image_train', 'dataset image_train'],
        'image': ['dataset-image', 'dataset image'],
        'cloth_train': ['dataset-cloth_train', 'dataset cloth_train'],
        'cloth': ['dataset-cloth', 'dataset cloth'],
        'parse': ['dataset-image-parse-v3', 'dataset image-parse-v3'],
        'cloth_mask': ['dataset-cloth-mask', 'dataset cloth-mask'],
        'openpose_img': ['dataset-openpose_img', 'dataset openpose_img'],
        'openpose_json': ['dataset-openpose_json', 'dataset openpose_json']
    },
    'preprocessing': {
        'image_size': [512, 384],
        'pose_keypoints': 18,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'cache_preprocessed': True,
        'cache_dir': '/content/cache'
    },
    'segmentation': {
        'model_type': 'transunet',
        'num_classes': 20,
        'in_channels': 3,
        'patch_size': 16,
        'num_layers': 12,
        'num_heads': 12,
        'hidden_size': 768,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'encoder_channels': [64, 128, 256, 512],
        'decoder_channels': [256, 128, 64, 32],
        'use_boundary_loss': True,
        'boundary_weight': 0.3,
        'dice_weight': 0.4,
        'ce_weight': 0.3
    },
    'warping': {
        'type': 'flow',
        'tps_points': 5,
        'refinement_stages': 3,
        'use_feedback': True
    },
    'training': {
        'batch_size': auto_config['batch_size'],
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'epochs': 50,
        'early_stopping_patience': 10,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 3,
        'use_amp': True,
        'amp_dtype': auto_config['amp_dtype'],
        'gradient_checkpointing': auto_config['gradient_checkpointing'],
        'gradient_accumulation_steps': auto_config['gradient_accumulation_steps'],
        'save_interval': 5,
        'checkpoint_dir': Path('/content/drive/MyDrive/vton_outputs/checkpoints')
    },
    'evaluation': {
        'metrics': ['miou', 'pixel_accuracy', 'ssim', 'lpips', 'fid'],
        'visualization_samples': 10,
        'save_predictions': True,
        'output_dir': Path('/content/drive/MyDrive/vton_outputs/results')
    }
}

# Save configuration
config_path = '/content/configs/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default=str)

print(f"Configuration saved to {config_path}")

# %% [markdown]
# ## 5. Download Sample VITON-HD Dataset

# %%
# Download sample VITON-HD data (for demonstration)
# In practice, use the full dataset

import gdown

# Sample dataset structure
sample_urls = {
    'sample_images': 'https://drive.google.com/uc?id=YOUR_SAMPLE_IMAGE_ID',
    'sample_cloth': 'https://drive.google.com/uc?id=YOUR_SAMPLE_CLOTH_ID',
    'sample_parse': 'https://drive.google.com/uc?id=YOUR_SAMPLE_PARSE_ID'
}

# Note: Replace with actual VITON-HD dataset URLs or mount from Drive
print("Please upload or mount the VITON-HD dataset to /content/")
print("Expected structure:")
print("  /content/dataset-image_train/")
print("  /content/dataset-cloth_train/")
print("  /content/dataset-image-parse-v3/")
print("  /content/dataset-openpose_img/")
print("  /content/dataset-openpose_json/")

# %% [markdown]
# ## 6. Load Modules

# %%
# Write module files
modules = {
    '/content/vton/__init__.py': '',
    '/content/vton/data/__init__.py': '',
    '/content/vton/models/__init__.py': '',
    '/content/vton/losses/__init__.py': '',
    '/content/vton/utils/__init__.py': '',
}

for filepath, content in modules.items():
    with open(filepath, 'w') as f:
        f.write(content)

# Add to Python path
sys.path.append('/content')

print("Modules initialized")

# %% [markdown]
# ## 7. Utility Functions

# %%
# Create utility functions
utils_metrics = '''
"""Metrics for evaluation"""
import torch
import numpy as np
from typing import Dict

class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict:
        """Compute segmentation metrics"""
        # Calculate IoU per class
        ious = []
        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                ious.append((intersection / union).item())
        
        # Mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        # Pixel Accuracy
        correct = (pred == target).sum().float()
        total = target.numel()
        pixel_accuracy = (correct / total).item()
        
        return {
            'miou': miou,
            'pixel_accuracy': pixel_accuracy,
            'per_class_iou': ious
        }
'''

with open('/content/vton/utils/metrics.py', 'w') as f:
    f.write(utils_metrics)

utils_checkpoint = '''
"""Checkpoint utilities"""
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})
'''

with open('/content/vton/utils/checkpoint.py', 'w') as f:
    f.write(utils_checkpoint)

utils_viz = '''
"""Visualization utilities"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(images, predictions, targets, num_samples=4):
    """Visualize segmentation results"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*4))
    
    for i in range(min(num_samples, len(images))):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.229 + 0.485).clip(0, 1)  # Denormalize
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Prediction
        pred = predictions[i].cpu().numpy()
        axes[i, 1].imshow(pred, cmap='tab20')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Target
        target = targets[i].cpu().numpy()
        axes[i, 2].imshow(target, cmap='tab20')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig
'''

with open('/content/vton/utils/visualization.py', 'w') as f:
    f.write(utils_viz)

print("Utility