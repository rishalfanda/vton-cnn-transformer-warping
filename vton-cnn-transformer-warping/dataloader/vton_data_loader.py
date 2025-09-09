"""
vton_data_loader.py
Dataset loader untuk VITON-like structure.

Usage:
  from dataloader.vton_data_loader import create_dataloaders, VITONDataset
  train_loader, val_loader = create_dataloaders(config)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import logging

logger = logging.getLogger("VITON-Dataloader")
logger.setLevel(logging.INFO)

# -------------------------
# Helper: pose json -> heatmaps
# -------------------------
def load_pose_json_to_heatmap(json_path: str, image_size: Tuple[int, int], num_keypoints: int = 18, sigma: int = 6):
    """
    Convert OpenPose JSON (COCO format) to heatmaps of shape (num_keypoints, H, W)
    - json_path: path to openpose json file
    - image_size: (H, W)
    """
    H, W = image_size
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    if not os.path.exists(json_path):
        return heatmaps

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read pose json {json_path}: {e}")
        return heatmaps

    people = data.get("people", [])
    if not people:
        return heatmaps

    keypoints = people[0].get("pose_keypoints_2d", [])
    if len(keypoints) < num_keypoints * 3:
        # malformed, return zeros
        return heatmaps

    for i in range(num_keypoints):
        x = keypoints[3 * i]
        y = keypoints[3 * i + 1]
        conf = keypoints[3 * i + 2]
        if conf <= 0.05:
            continue
        # integer coordinates clamp
        cx = int(np.clip(x, 0, W - 1))
        cy = int(np.clip(y, 0, H - 1))
        # gaussian heatmap
        tmp = np.zeros((H, W), dtype=np.float32)
        # make grid
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        exponent = ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (sigma ** 2))
        tmp = np.exp(-exponent)
        tmp = tmp / (tmp.max() + 1e-8)
        heatmaps[i] = tmp

    return heatmaps


# -------------------------
# Default transforms (albumentations)
# -------------------------
def get_transforms(config: Dict, mode: str = "train"):
    H, W = config["preprocessing"]["image_size"]
    mean = config["preprocessing"]["normalize_mean"]
    std = config["preprocessing"]["normalize_std"]

    if mode == "train":
        aug = A.Compose([
            A.Resize(height=H, width=W),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        aug = A.Compose([
            A.Resize(height=H, width=W),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    return aug


# -------------------------
# VITON Dataset class
# -------------------------
class VITONDataset(Dataset):
    def __init__(self, data_root: str, mode: str = "train", config: Dict = None, transform=None):
        self.root = Path(data_root)
        self.mode = mode
        self.config = config or {}
        self.transform = transform or get_transforms(self.config, mode)
        self.aliases = self.config.get("dataset_aliases", {})
        self.cache_enabled = self.config.get("preprocessing", {}).get("cache_preprocessed", False)
        self.cache_dir = Path(self.config.get("preprocessing", {}).get("cache_dir", "/content/cache"))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # build file lists
        self.data_list = self._build_data_list()
        logger.info(f"VITONDataset mode={mode} loaded {len(self.data_list)} samples from {data_root}")

    def _find_folder(self, alias_list: List[str]) -> Optional[Path]:
        for alias in alias_list:
            # try variations
            candidates = [
                alias,
                alias.replace("-", "_"),
                alias.replace("_", "-"),
                os.path.join(alias),
            ]
            for c in candidates:
                p = self.root / c
                if p.exists():
                    return p
        return None

    def _build_data_list(self) -> List[dict]:
        # determine which aliases to use depending on mode
        aliases = self.aliases
        if self.mode == "train":
            img_alias = aliases.get("image_train", [])
            cloth_alias = aliases.get("cloth_train", [])
        else:
            img_alias = aliases.get("image", [])
            cloth_alias = aliases.get("cloth", [])

        image_dir = self._find_folder(img_alias)
        cloth_dir = self._find_folder(cloth_alias)
        parse_dir = self._find_folder(self.aliases.get("parse", []))
        pose_json_dir = self._find_folder(self.aliases.get("openpose_json", []))
        cloth_mask_dir = self._find_folder(self.aliases.get("cloth_mask", []))

        if image_dir is None or cloth_dir is None:
            raise ValueError(f"Required dataset folders not found. image_dir={image_dir} cloth_dir={cloth_dir}")

        # if pairs file exists (train_pairs.txt), use it; otherwise pair by same index/name
        pairs_file = self.root / f"{self.mode}_pairs.txt"
        pairs = []
        if pairs_file.exists():
            with open(pairs_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pairs.append((parts[0], parts[1]))
        else:
            img_files = sorted([p.name for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".png"]])
            cloth_files = sorted([p.name for p in cloth_dir.glob("*") if p.suffix.lower() in [".jpg", ".png"]])
            # simple pairing: pair same-index (or pair each image with first cloth to avoid huge pairing)
            min_len = min(len(img_files), len(cloth_files))
            for i in range(min_len):
                pairs.append((img_files[i], cloth_files[i]))

        data_list = []
        for img_name, cloth_name in pairs:
            item = {
                "person_img": image_dir / img_name,
                "cloth_img": cloth_dir / cloth_name
            }
            person_id = Path(img_name).stem
            cloth_id = Path(cloth_name).stem

            # optional files
            if parse_dir:
                fp = parse_dir / f"{person_id}.png"
                if fp.exists(): item["parse"] = fp
            if pose_json_dir:
                pj = pose_json_dir / f"{person_id}_keypoints.json"
                if pj.exists(): item["pose_json"] = pj
            if cloth_mask_dir:
                cm = cloth_mask_dir / f"{cloth_id}.jpg"
                if cm.exists(): item["cloth_mask"] = cm

            # validate existence
            if not item["person_img"].exists() or not item["cloth_img"].exists():
                continue
            data_list.append(item)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, p: Path):
        img = Image.open(p).convert("RGB")
        img = np.array(img)
        return img

    def _load_parse(self, p: Path):
        # parse maps assumed PNG with integer labels
        img = Image.open(p).convert("L")
        arr = np.array(img).astype(np.int64)
        return arr

    def __getitem__(self, idx):
        item = self.data_list[idx]
        person_img = self._load_image(item["person_img"])
        cloth_img = self._load_image(item["cloth_img"])

        # optional masks/parses
        parse = None
        if "parse" in item:
            parse = self._load_parse(item["parse"])

        # pose heatmap
        pose_hm = None
        if "pose_json" in item:
            H, W = self.config["preprocessing"]["image_size"]
            pose_hm = load_pose_json_to_heatmap(str(item["pose_json"]), image_size=(H, W),
                                                num_keypoints=self.config["preprocessing"].get("pose_keypoints", 18),
                                                sigma=6)

        # apply transform (albumentations expects dict keys like image, mask)
        if self.transform:
            # If parse exists, pass as mask to keep same transform
            if parse is not None:
                transformed = self.transform(image=person_img, mask=parse)
                person_t = transformed["image"]
                parse_t = transformed["mask"]
            else:
                transformed = self.transform(image=person_img)
                person_t = transformed["image"]
                parse_t = None

            # cloth transform (resize + normalize) - reuse same transform but apply on cloth
            cloth_trans = get_transforms(self.config, mode="train" if self.mode=="train" else "val")
            cloth_transformed = cloth_trans(image=cloth_img)
            cloth_t = cloth_transformed["image"]
        else:
            # fallback: to tensor manually
            person_t = torch.from_numpy(person_img).permute(2,0,1).float() / 255.0
            cloth_t = torch.from_numpy(cloth_img).permute(2,0,1).float() / 255.0
            parse_t = torch.from_numpy(parse).long() if parse is not None else None

        result = {
            "person_img": person_t,
            "cloth_img": cloth_t,
            "person_id": item["person_img"].stem,
            "cloth_id": item["cloth_img"].stem
        }

        if parse_t is not None:
            # if albumentations returned numpy mask, convert to long tensor
            if isinstance(parse_t, np.ndarray):
                parse_t = torch.from_numpy(parse_t).long()
            result["parse"] = parse_t

        if pose_hm is not None:
            # pose_hm shape: (K,H,W) -> convert to torch float tensor
            result["pose"] = torch.from_numpy(pose_hm).float()

        if "cloth_mask" in item:
            cm = Image.open(item["cloth_mask"]).convert("L")
            cm = cm.resize((self.config["preprocessing"]["image_size"][1], self.config["preprocessing"]["image_size"][0]))
            cm = np.array(cm)
            cm = torch.from_numpy(cm).unsqueeze(0).float() / 255.0
            result["cloth_mask"] = cm

        return result


# -------------------------
# create_dataloaders helper
# -------------------------
def create_dataloaders(config: Dict, batch_size_override: Optional[int] = None):
    data_root = config["dataset"]["data_root"]
    batch_size = batch_size_override or config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    pin_memory = config["dataset"].get("pin_memory", True)

    train_ds = VITONDataset(data_root=data_root, mode="train", config=config, transform=get_transforms(config, "train"))
    val_ds = VITONDataset(data_root=data_root, mode="test", config=config, transform=get_transforms(config, "val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers//2), pin_memory=pin_memory)
    return train_loader, val_loader


# -------------------------
# Quick test function (run in notebook)
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/content/vton_project/configs/vton_config_default.py")
    args = parser.parse_args()
    # dynamic import config
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", args.config)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = cfg_mod.config

    tr, va = create_dataloaders(cfg, batch_size_override=2)
    batch = next(iter(tr))
    print("Keys in batch:", batch.keys())
    print("person_img shape:", batch["person_img"].shape)
    if "parse" in batch:
        print("parse shape:", batch["parse"].shape)
    if "pose" in batch:
        print("pose shape:", batch["pose"].shape)
