import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import PIL
import skimage.io as io

class MedicalDataset(Dataset):
    def __init__(self, 
        dataset_name,
        dataset_dir, 
        data_list, 
        transforms: Optional[Callable] = None,
        use_ori_mask = False
        ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.data_list = data_list
        self.transforms = transforms
        self.use_ori_mask = use_ori_mask

    def __len__(self,) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        try:
            image = Image.open(os.path.join(self.dataset_dir, self.data_list[index]['image'])).convert('RGB') # for SZ-CXR
            mask = Image.open(os.path.join(self.dataset_dir, self.data_list[index]['label'])).convert('L')
            assert image.size == mask.size

        except PIL.UnidentifiedImageError:
            image = io.imread(os.path.join(self.dataset_dir, self.data_list[index]['image']))
            mask = io.imread(os.path.join(self.dataset_dir, self.data_list[index]['label']))
            assert image.shape[:2] == mask.shape[:2]

        if self.transforms is not None:
            image_arr = np.array(image)
            mask_arr = np.array(mask)
            if mask_arr.max() == 255:
                mask_arr = mask_arr/255
            mask_arr = mask_arr.astype(np.uint8)
            
            ori_mask = mask_arr
            transformed = self.transforms(image=image_arr, mask=mask_arr)
            image = transformed['image']
            mask = transformed['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).float()
        
        data_item = {'image': image, 'label': mask}

        if self.dataset_name == 'Polyp':
            dataset_name = self.data_list[index]['label'].split('/')[2] # fixed.
        else:
            dataset_name = self.dataset_name
        data_item['dataset_name'] = dataset_name

        data_name = self.data_list[index]['image'].split('/')[-1]
        data_item['data_name'] = data_name

        if self.use_ori_mask:
            data_item['ori_mask'] = ori_mask
        
        return data_item

class MedicalSegDataModule(pl.LightningDataModule):
    def __init__(self, 
        dataset_name: Optional[str] = None, 
        dataset_dir: Optional[str] = None, 
        data_info_dir: Optional[str] = None, 
        batch_size = 2,
        image_size: int = 256, 
        num_workers: int = 1,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        pred_transforms: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.dataset_name = dataset_name
        self.data_info_dir = data_info_dir
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.data_info = os.path.join(data_info_dir, f"{dataset_name}_dataset_info.json")

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_workers = num_workers

        self.train_transforms = train_transforms or self._default_train_transforms()
        self.val_transforms = val_transforms or self._default_transforms()
        self.test_transforms = test_transforms or self._default_transforms()
        self.pred_transforms = pred_transforms or self._default_transforms()

        self.data_info = self.parse_json(self.data_info)

    @staticmethod
    def parse_json(json_path):
        with open(json_path, 'r') as f:
            data_info = json.load(f)
        return data_info

    def train_dataloader(self) -> DataLoader:
        
        dataset = MedicalDataset(
            self.dataset_name, 
            self.dataset_dir, 
            data_list=self.data_info['train'], 
            transforms=self.train_transforms)
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        
        dataset = MedicalDataset(
            self.dataset_name, 
            self.dataset_dir, 
            data_list=self.data_info['val'], 
            transforms=self.val_transforms,
            use_ori_mask=False)
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader
    
    def test_dataloader(self) -> DataLoader:
        
        dataset = MedicalDataset(
            self.dataset_name, 
            self.dataset_dir, 
            data_list=self.data_info['test'], 
            transforms=self.test_transforms, 
            use_ori_mask=True)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self) -> DataLoader:
        
        dataset = MedicalDataset(
            self.dataset_name, 
            self.dataset_dir, 
            data_list=self.data_info['test'], 
            transforms=self.pred_transforms, 
            use_ori_mask=True)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,)
        
        return loader


    def _default_transforms(self) -> Callable:
        transforms = A.Compose(
            [   
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        return transforms
    

    def _default_train_transforms(self) -> Callable:
        train_transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(self.image_size, self.image_size),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
                ),
            ToTensorV2()
            ])
        return train_transforms
    

