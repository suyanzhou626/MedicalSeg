f""" Version 3
We add the multi-scale input.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from msdataloader import MedicalSegDataModule
from pytorch_lightning.cli import LightningCLI
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils import binary_dice_metric, threshold_binary_dice_metrics
from model.fpn_pvt import FPN_PVT
from archs import NestedUNet, UNet
import os
import cv2
from PIL import Image
import pandas as pd
from loss import structure_loss

class SegModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, predict_upsample = False):
        super().__init__()

        self.lr = learning_rate
        self.model_name = model_name
        self.predict_upsample = predict_upsample

        self.net = eval(model_name)(num_classes=1, input_channels=3)

        self.loss_func = structure_loss
        # self.loss_func = F.binary_cross_entropy_with_logits

        self.test_result = {}

    def forward(self, x):
        return self.net(x)
    

    def loss(self, logits, labels):
        return self.loss_func(logits, labels)
    

    def training_step(self, batch, batch_idx):

        img, seg = batch['image'], batch['label']
        logits = self.forward(img)
        loss = self.loss(logits, seg)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        img, seg = batch['image'], batch['label']
        logits = self.forward(img)
        loss = self.loss(logits, seg)
        # dice = threshold_binary_dice_metrics(logits, ori_mask, up_mode=True)
        dice = binary_dice_metric(logits, seg)

        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('dice', dice.item(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        img, seg = batch['image'], batch['label']
        ori_mask = batch['ori_mask']
        dataset_name = batch['dataset_name'][0]

        logits = self.forward(img)

        if dataset_name not in self.test_result:
            self.test_result[dataset_name] = []
        
        thres_dice = threshold_binary_dice_metrics(logits, ori_mask, up_mode=self.predict_upsample)
        self.test_result[dataset_name].append(thres_dice.item())

    def test_epoch_end(self, outputs):

        result = {}
        for dataset in self.test_result.keys():
            result[dataset] =  np.mean(self.test_result[dataset])

        pd.DataFrame.from_dict(result, orient='index', columns=['mDice'])\
            .to_csv(os.path.join(self.trainer.log_dir, 'result.csv'))

        mean_thres_dice = np.mean(list(result.values()))

        self.log('test_thres_dice', mean_thres_dice, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):

        img = batch['image']
        dataset_name = batch['dataset_name'][0]
        data_name = batch['data_name'][0]
        ori_mask = batch['ori_mask']
        logits = self.forward(img)

        if self.predict_upsample:
            logits = F.interpolate(logits, size=ori_mask.shape[-2:], \
                mode='bilinear', align_corners=False)

        logits = logits.sigmoid().squeeze().data.cpu().numpy()
        logits = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
        logits = (logits * 255).astype(np.uint8)
        
        # predict and save
        save_dir = os.path.join(self.trainer.log_dir, 'result_map', dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        Image.fromarray(logits).save(os.path.join(save_dir, data_name))

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = optim.AdamW(trainable_parameters, lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=80, gamma=0.5)
        return [optimizer], [scheduler]


def cli_main():
    cli = LightningCLI(datamodule_class=MedicalSegDataModule, model_class=SegModel, seed_everything_default=False, run=False)  
   
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")
    cli.trainer.predict(cli.model, cli.datamodule, ckpt_path="best")




if __name__ == "__main__":

    cli_main()
    