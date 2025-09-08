# VTON Pipeline Configuration
# Sesuai Bab IV Tesis - Metodologi Penelitian

dataset:
  data_root: "/content"
  subset: "all"
  train_size: 0.8
  val_size: 0.2
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

dataset_aliases:
  image_train: ["dataset-image_train", "dataset image_train"]
  image: ["dataset-image", "dataset image"]
  cloth_train: ["dataset-cloth_train", "dataset cloth_train"]
  cloth: ["dataset-cloth", "dataset cloth"]
  parse: ["dataset-image-parse-v3", "dataset image-parse-v3", "dataset-image-parse-v3-test"]
  cloth_mask: ["dataset-cloth-mask", "dataset cloth-mask", "dataset-cloth-mask-test"]
  openpose_img: ["dataset-openpose_img", "dataset openpose_img", "dataset-openpose_img-test"]
  openpose_json: ["dataset-openpose_json", "dataset openpose_json", "dataset-openpose_json-test"]

preprocessing:
  image_size: [512, 384]  # height, width untuk VITON-HD
  pose_keypoints: 18
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  cache_preprocessed: true
  cache_dir: "/content/cache"

segmentation:
  # Bab IV 4.3.2 - Model Segmentasi
  model_type: "transunet"  # ["unet", "transunet"]
  num_classes: 20  # Sesuai VITON-HD parsing labels
  in_channels: 3
  
  # TransUNet specific
  patch_size: 16
  num_layers: 12
  num_heads: 12
  hidden_size: 768
  mlp_dim: 3072
  dropout: 0.1
  
  # U-Net backbone
  encoder_channels: [64, 128, 256, 512]
  decoder_channels: [256, 128, 64, 32]
  
  # Bab IV 4.3.3 - Boundary-aware loss
  use_boundary_loss: true
  boundary_weight: 0.3
  dice_weight: 0.4
  ce_weight: 0.3

warping:
  # Bab IV 4.3.4 - Warping Module
  type: "tps"  # ["tps", "flow", "neural"]
  tps_points: 5
  refinement_stages: 3
  use_feedback: true
  
training:
  # Training hyperparameters
  batch_size: 8  # akan di-auto adjust
  learning_rate: 0.0001
  weight_decay: 0.0001
  epochs: 100
  early_stopping_patience: 15
  
  # Optimizer
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 5
  
  # Mixed precision & memory optimization
  use_amp: true
  amp_dtype: "fp16"  # ["fp16", "bf16"]
  gradient_checkpointing: true
  gradient_accumulation_steps: 1
  
  # Checkpointing
  save_interval: 5
  checkpoint_dir: "/content/drive/MyDrive/vton_outputs/checkpoints"
  
evaluation:
  # Bab IV 4.4 - Metrik Evaluasi
  metrics: ["miou", "pixel_accuracy", "ssim", "lpips", "fid"]
  visualization_samples: 10
  save_predictions: true
  output_dir: "/content/drive/MyDrive/vton_outputs/results"

experiments:
  # Bab IV 4.5 - Skenario Eksperimen
  baseline:
    model: "unet"
    use_boundary_loss: false
    
  proposed:
    model: "transunet"
    use_boundary_loss: true
    
  ablation:
    transunet_no_boundary:
      model: "transunet"
      use_boundary_loss: false
    unet_with_boundary:
      model: "unet"
      use_boundary_loss: true

# Auto-configuration based on GPU
auto_config:
  detect_gpu: true
  gpu_configs:
    T4:
      batch_size: 4
      gradient_accumulation_steps: 2
      amp_dtype: "fp16"
    P100:
      batch_size: 6
      gradient_accumulation_steps: 1
      amp_dtype: "fp16"
    V100:
      batch_size: 8
      gradient_accumulation_steps: 1
      amp_dtype: "fp16"
    A100:
      batch_size: 16
      gradient_accumulation_steps: 1
      amp_dtype: "bf16"