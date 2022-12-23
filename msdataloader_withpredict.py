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
from msdataloader import MedicalSegDataModule
from msdataloader import MedicalDataset

class MedicalSegWPredictDataModule(MedicalSegDataModule):
    def __init__(self, 
        # dataset_name: Optional[str] = None, 
        # dataset_dir: Optional[str] = None, 
        # data_info_dir: Optional[str] = None, 
        # batch_size = 2,
        # image_size: int = 256, 
        # num_workers: int = 1,
        # shuffle: bool = True,
        # pin_memory: bool = True,
        # drop_last: bool = False,
        # train_transforms: Optional[Callable] = None,
        # val_transforms: Optional[Callable] = None,
        # test_transforms: Optional[Callable] = None,
        pred_transforms: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.pred_transforms = pred_transforms or self._default_transforms

    # @staticmethod
    # def parse_json(json_path):
    #     with open(json_path, 'r') as f:
    #         data_info = json.load(f)
    #     return data_info

    # def train_dataloader(self) -> DataLoader:
        
    #     dataset = MedicalDataset(self.dataset_dir, data_list=self.data_info['train'], transforms=self.train_transforms)
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=self.shuffle,
    #         num_workers=self.num_workers,
    #         drop_last=self.drop_last,
    #         pin_memory=self.pin_memory,
    #     )
    #     return loader

    # def val_dataloader(self) -> DataLoader:
        
    #     dataset = MedicalDataset(self.dataset_dir, data_list=self.data_info['val'], transforms=self.val_transforms)
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         drop_last=self.drop_last,
    #         pin_memory=self.pin_memory,
    #     )
    #     return loader
    
    # def test_dataloader(self) -> DataLoader:
        
    #     dataset = MedicalDataset(self.dataset_dir, data_list=self.data_info['test'], transforms=self.test_transforms)
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=8,
    #         drop_last=False,
    #         pin_memory=True,
    #     )
    #     return loader

    def predict_dataloader(self) -> DataLoader:
        
        dataset = MedicalDataset(self.dataset_name, self.dataset_dir, data_list=self.data_info['test'], transforms=self.test_transforms, predict_phase=True)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,)
        
        return loader


