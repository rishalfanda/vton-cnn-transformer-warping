"""
VTON Dataset Loader
Bab IV 4.2 - Dataset dan Preprocessing
Implementasi data loading untuk VITON-HD dengan support alias folder
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VITONDataset(Dataset):
    """
    Dataset class untuk VITON-HD
    Bab IV 4.2.1 - Dataset VITON-HD
    """
    
    def __init__(
        self,
        data_root: str,
        mode: str = 'train',
        config: Dict = None,
        transform: Optional[transforms.Compose] = None
    ):
        self.data_root = Path(data_root)
        self.mode = mode
        self.config = config or {}
        self.transform = transform or self._get_default_transform()
        
        # Dataset aliases untuk fleksibilitas naming
        self.aliases = config.get('dataset_aliases', {})
        
        # Caching
        self.cache_dir = Path(config.get('preprocessing', {}).get('cache_dir', '/content/cache'))
        self.cache_enabled = config.get('preprocessing', {}).get('cache_preprocessed', True)
        
        # Load data paths
        self.data_list = self._load_data_list()
        
        logger.info(f"Loaded {len(self.data_list)} samples for {mode} mode")
    
    def _find_folder(self, folder_aliases: List[str]) -> Optional[Path]:
        """Cari folder dengan berbagai kemungkinan nama"""
        for alias in folder_aliases:
            # Coba dengan dan tanpa underscore/dash
            variations = [
                alias,
                alias.replace('-', '_'),
                alias.replace('_', '-'),
                alias.replace('-', ' '),
                alias.replace('_', ' ')
            ]
            
            for var in variations:
                path = self.data_root / var
                if path.exists():
                    logger.debug(f"Found folder: {path}")
                    return path
        
        logger.warning(f"No folder found for aliases: {folder_aliases}")
        return None
    
    def _load_data_list(self) -> List[Dict]:
        """Load dan validasi data paths"""
        data_list = []
        
        # Tentukan folder berdasarkan mode
        if self.mode == 'train':
            image_aliases = self.aliases.get('image_train', ['dataset-image_train'])
            cloth_aliases = self.aliases.get('cloth_train', ['dataset-cloth_train'])
        else:
            image_aliases = self.aliases.get('image', ['dataset-image'])
            cloth_aliases = self.aliases.get('cloth', ['dataset-cloth'])
        
        # Find actual folders
        image_dir = self._find_folder(image_aliases)
        cloth_dir = self._find_folder(cloth_aliases)
        parse_dir = self._find_folder(self.aliases.get('parse', ['dataset-image-parse-v3']))
        pose_img_dir = self._find_folder(self.aliases.get('openpose_img', ['dataset-openpose_img']))
        pose_json_dir = self._find_folder(self.aliases.get('openpose_json', ['dataset-openpose_json']))
        cloth_mask_dir = self._find_folder(self.aliases.get('cloth_mask', ['dataset-cloth-mask']))
        
        if not image_dir or not cloth_dir:
            raise ValueError(f"Required directories not found. Image: {image_dir}, Cloth: {cloth_dir}")
        
        # Load pairs file jika ada
        pairs_file = self.data_root / f"{self.mode}_pairs.txt"
        if pairs_file.exists():
            with open(pairs_file, 'r') as f:
                pairs = [line.strip().split() for line in f]
        else:
            # Generate pairs dari available files
            image_files = sorted([f.name for f in image_dir.glob("*.jpg")])
            cloth_files = sorted([f.name for f in cloth_dir.glob("*.jpg")])
            pairs = [(img, cloth) for img in image_files for cloth in cloth_files[:1]]  # Simple pairing
        
        # Validate dan build data list
        for person_img, cloth_img in pairs:
            person_id = person_img.replace('.jpg', '')
            cloth_id = cloth_img.replace('.jpg', '')
            
            data_item = {
                'person_img': image_dir / person_img,
                'cloth_img': cloth_dir / cloth_img,
                'person_id': person_id,
                'cloth_id': cloth_id
            }
            
            # Optional files
            if parse_dir:
                parse_file = parse_dir / f"{person_id}.png"
                if parse_file.exists():
                    data_item['parse'] = parse_file
            
            if pose_img_dir:
                pose_img = pose_img_dir / f"{person_id}_rendered.png"
                if pose_img.exists():
                    data_item['pose_img'] = pose_img
            
            if pose_json_dir:
                pose_json = pose_json_dir / f"{person_id}_keypoints.json"
                if pose_json.exists():
                    data_item['pose_json'] = pose_json
            
            if cloth_mask_dir:
                cloth_mask = cloth_mask_dir / f"{cloth_id}.jpg"
                if cloth_mask.exists():
                    data_item['cloth_mask'] = cloth_mask
            
            # Skip incomplete data if required
            if self._validate_data_item(data_item):
                data_list.append(data_item)
        
        return data_list
    
    def _validate_data_item(self, item: Dict) -> bool:
        """Validasi kelengkapan data item"""
        required = ['person_img', 'cloth_img']
        for key in required:
            if key not in item or not item[key].exists():
                return False
        return True
    
    def _get_default_transform(self) -> transforms.Compose:
        """Default transformation pipeline"""
        img_size = self.config.get('preprocessing', {}).get('image_size', [512, 384])
        mean = self.config.get('preprocessing', {}).get('normalize_mean', [0.485, 0.456, 0.406])
        std = self.config.get('preprocessing', {}).get('normalize_std', [0.229, 0.224, 0.225])
        
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image with caching support"""
        if self.cache_enabled:
            cache_path = self.cache_dir / f"{path.stem}_cached.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        
        img = Image.open(path).convert('RGB')
        return img
    
    def _load_pose(self, json_path: Path) -> np.ndarray:
        """Load pose keypoints dari JSON"""
        with open(json_path, 'r') as f:
            pose_data = json.load(f)
        
        # Extract keypoints (format COCO)
        keypoints = pose_data['people'][0]['pose_keypoints_2d'] if pose_data['people'] else []
        keypoints = np.array(keypoints).reshape(-1, 3)  # x, y, confidence
        
        return keypoints
    
    def _load_parse(self, parse_path: Path) -> torch.Tensor:
        """Load segmentation parse map"""
        parse = Image.open(parse_path)
        parse = np.array(parse)
        
        # Convert to tensor
        parse_tensor = torch.from_numpy(parse).long()
        
        return parse_tensor
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get data item
        Returns dictionary dengan keys:
        - person_img: transformed person image
        - cloth_img: transformed cloth image
        - parse: segmentation map (if available)
        - pose: pose keypoints (if available)
        - cloth_mask: cloth mask (if available)
        - meta: metadata
        """
        data_item = self.data_list[idx]
        
        # Load images
        person_img = self._load_image(data_item['person_img'])
        cloth_img = self._load_image(data_item['cloth_img'])
        
        # Apply transforms
        if self.transform:
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)
        
        result = {
            'person_img': person_img,
            'cloth_img': cloth_img,
            'person_id': data_item['person_id'],
            'cloth_id': data_item['cloth_id']
        }
        
        # Load optional data
        if 'parse' in data_item:
            result['parse'] = self._load_parse(data_item['parse'])
        
        if 'pose_json' in data_item:
            result['pose'] = torch.from_numpy(self._load_pose(data_item['pose_json'])).float()
        
        if 'cloth_mask' in data_item:
            cloth_mask = Image.open(data_item['cloth_mask']).convert('L')
            cloth_mask = transforms.ToTensor()(cloth_mask)
            result['cloth_mask'] = cloth_mask
        
        return result


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    Bab IV 4.2.2 - Data Pipeline
    """
    # Create datasets
    train_dataset = VITONDataset(
        data_root=config['dataset']['data_root'],
        mode='train',
        config=config
    )
    
    val_dataset = VITONDataset(
        data_root=config['dataset']['data_root'],
        mode='test',
        config=config
    )
    
    # Create dataloaders dengan optimasi memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=config['dataset'].get('pin_memory', True),
        prefetch_factor=config['dataset'].get('prefetch_factor', 2),
        persistent_workers=True if config['dataset'].get('num_workers', 4) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=config['dataset'].get('pin_memory', True)
    )
    
    return train_loader, val_loader