"""
HashCenterDataset

PyTorch Dataset class that extends ImageList with hash center assignment.
"""

import torch
import numpy as np
import logging
from typing import Optional, Callable, Tuple, Union
from PIL import Image
from .generator import HashCenterGenerator
from .calculator import CentroidCalculator

logger = logging.getLogger(__name__)


def _pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def _make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


class HashCenterDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that extends ImageList with hash center assignment.
    
    Maintains backward compatibility: when hash center parameters are not
    provided, behaves identically to ImageList.
    """
    
    def __init__(self, 
                 image_list: list,
                 labels: Optional[np.ndarray] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable = _pil_loader,
                 # Hash center parameters
                 num_classes: Optional[int] = None,
                 hash_bit: Optional[int] = None,
                 hash_method: str = 'hadamard',
                 dataset_type: str = 'auto',
                 dataset_name: Optional[str] = None,
                 hash_centers_path: Optional[str] = None,
                 enable_hash_centers: bool = False,
                 validate_hash_centers: bool = True,
                 save_hash_centers: bool = False,
                 save_path: Optional[str] = None):
        if enable_hash_centers:
            if num_classes is None or hash_bit is None:
                raise ValueError(
                    "num_classes and hash_bit are required when enable_hash_centers=True"
                )
            if save_hash_centers and save_path is None:
                raise ValueError(
                    "save_path is required when save_hash_centers=True"
                )
        
        self.imgs = _make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image_list")
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.enable_hash_centers = enable_hash_centers
        self.num_classes = num_classes
        self.hash_bit = hash_bit
        self.hash_method = hash_method
        self.dataset_name = dataset_name
        self.validate_hash_centers = validate_hash_centers
        self.save_hash_centers = save_hash_centers
        self.save_path = save_path
        self.hash_centers_path = hash_centers_path
        
        self.class_hash_centers = None
        self.dataset_type = None
        
        if self.enable_hash_centers:
            if dataset_type == 'auto':
                self.dataset_type = self._detect_dataset_type()
            else:
                self.dataset_type = dataset_type
            
            logger.info(f"Dataset type detected: {self.dataset_type}")
            self._initialize_hash_centers()
    
    def _detect_dataset_type(self) -> str:
        _, first_label = self.imgs[0]
        if isinstance(first_label, (int, np.integer)):
            return 'single-label'
        elif isinstance(first_label, (np.ndarray, torch.Tensor, list)):
            return 'multi-label'
        else:
            logger.warning(
                f"Unable to auto-detect dataset type from label type {type(first_label)}. "
                f"Defaulting to 'single-label'"
            )
            return 'single-label'
    
    def _initialize_hash_centers(self):
        generator = HashCenterGenerator(
            num_classes=self.num_classes,
            hash_bit=self.hash_bit,
            method=self.hash_method
        )
        
        if self.hash_centers_path is not None:
            logger.info(f"Loading hash centers from {self.hash_centers_path}")
            self.class_hash_centers = generator.load(self.hash_centers_path)
            
            if self.validate_hash_centers:
                from .validator import HashCenterValidator
                HashCenterValidator.validate(self.class_hash_centers, log_stats=True)
        else:
            logger.info(f"Generating hash centers using {self.hash_method} method")
            self.class_hash_centers = generator.generate(
                dataset_name=self.dataset_name,
                validate=self.validate_hash_centers,
                use_cache=True
            )
            
            if self.save_hash_centers:
                logger.info(f"Saving hash centers to {self.save_path}")
                metadata = {
                    'dataset_name': self.dataset_name,
                    'dataset_type': self.dataset_type
                }
                generator.save(self.save_path, metadata=metadata)
    
    def _get_hash_center_for_sample(self, label: Union[int, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.dataset_type == 'single-label':
            if not isinstance(label, (int, np.integer)):
                label = int(label)
            if label < 0 or label >= self.num_classes:
                raise IndexError(
                    f"Class index {label} out of range for {self.num_classes} classes"
                )
            return self.class_hash_centers[label]
        
        elif self.dataset_type == 'multi-label':
            if isinstance(label, np.ndarray):
                label_vector = torch.from_numpy(label).float()
            elif isinstance(label, list):
                label_vector = torch.tensor(label, dtype=torch.float32)
            else:
                label_vector = label.float()
            return CentroidCalculator.calculate(label_vector, self.class_hash_centers)
        
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def __getitem__(self, index: int):
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if not self.enable_hash_centers:
            return img, target
        else:
            hash_center = self._get_hash_center_for_sample(target)
            return img, target, hash_center
    
    def __len__(self) -> int:
        return len(self.imgs)
