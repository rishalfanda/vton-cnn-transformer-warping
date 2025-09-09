"""
vton_config_default.py
Konfigurasi utama untuk eksperimen VTON (format: python dict)
Sesuaikan `dataset.data_root` ke lokasi dataset Anda di Google Drive.
"""

from pathlib import Path

ROOT = "/content"  # ubah jika perlu

config = {
    # ---------- Dataset ----------
    "dataset": {
        # ganti path ini sesuai lokasi dataset di Drive Anda
        "data_root": str(Path(ROOT) / "vton_dataset"),
        "mode": "train",
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 2
    },

    # folder name aliases untuk fleksibilitas struktur dataset
    "dataset_aliases": {
        "image_train": ["image_train", "image/image_train"],
        "image": ["image_test", "image/image_test"],

        "cloth_train": ["cloth_train", "cloth/cloth_train"],
        "cloth": ["cloth_test", "cloth/cloth_test"],

        "parse": [
            "image_parse_train", "image_parse/image_parse_train",
            "image_parse_test", "image_parse/image_parse_test"
        ],

        "cloth_mask": [
            "cloth_mask_train", "cloth_mask/cloth_mask_train",
            "cloth_mask_test", "cloth_mask/cloth_mask_test"
        ],

        "openpose_img": [
            "openpose_img_train", "openpose_img/openpose_img_train",
            "openpose_img_test", "openpose_img/openpose_img_test"
        ],

        "openpose_json": [
            "openpose_json_train", "openpose_json/openpose_json_train",
            "openpose_json_test", "openpose_json/openpose_json_test"
        ]
    },

    # ---------- Preprocessing ----------
    "preprocessing": {
        # ukuran H, W (pilih 512x384 untuk training awal; 1024x768 untuk final jika resource cukup)
        "image_size": [512, 384],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "cache_preprocessed": True,
        "cache_dir": "/content/cache",
        "pose_keypoints": 18  # jumlah keypoints OpenPose (COCO)
    },

    # ---------- Segmentation model ----------
    "segmentation": {
        "in_channels": 3,
        "num_classes": 20,  # sesuaikan dengan jumlah label parse Anda
        "model_type": "transunet",   # default (bisa overriden oleh --exp)
        "encoder_channels": [64, 128, 256, 512],
        "decoder_channels": [256, 128, 64, 32],
        "hidden_size": 768,
        "num_layers": 8,
        "num_heads": 8,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "patch_size": 16,

        # loss weights (default)
        "dice_weight": 0.4,
        "ce_weight": 0.3,
        "boundary_weight": 0.3,
        "use_boundary_loss": True
    },

    # ---------- Warping ----------
    "warping": {
        "type": "tps",     # pilih "tps" (kita gunakan TPS untuk semua eksperimen)
        "tps_points": 5,
        "refinement_stages": 3,
        "use_feedback": True
    },

    # ---------- Training ----------
    "training": {
        "epochs": 50,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "gradient_checkpointing": False,
        "use_amp": False,
        "save_interval": 5,
        # default checkpoint dir (akan diganti per-experiment oleh script)
        "checkpoint_dir": str(Path(ROOT) / "vton_outputs" / "checkpoints")
    },

    # ---------- Evaluation / output ----------
    "evaluation": {
        # default results dir (akan diganti per-experiment oleh script)
        "output_dir": str(Path(ROOT) / "vton_outputs" / "results"),
        "metrics": ["miou", "pixel_accuracy", "fid", "lpips", "ssim"],
        "save_visuals": True
    },

    # ---------- Experiments mapping ----------
    "experiments": {
        # simple mapping to change model_type and use_boundary_loss
        "baseline": {"model": "unet", "use_boundary_loss": False},
        "proposed": {"model": "transunet", "use_boundary_loss": True},
        "ablation": {
            # two ablation variants accessible via names:
            "transunet_no_boundary": {"model": "transunet", "use_boundary_loss": False},
            "unet_with_boundary": {"model": "unet", "use_boundary_loss": True}
        }
    }
}
