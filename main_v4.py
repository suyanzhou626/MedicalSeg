f""" Version 3
We add the inference code.
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
import pandas as pd
from loss import structure_loss

class SegModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, pretrained_model = None):
        super().__init__()

        self.lr = learning_rate
        self.model_name = model_name
        self.net = eval(model_name)(num_classes=1, input_channels=3)
        # self.loss_func = F.binary_cross_entropy_with_logits
        self.loss_func = structure_loss

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
        dice = binary_dice_metric(logits, seg)

        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('dice', dice.item(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        img, seg = batch['image'], batch['label']
        ori_mask = batch['ori_mask']
        logits = self.forward(img)
        dice = binary_dice_metric(logits, seg, threshold=0.5)
        thres_dice = threshold_binary_dice_metrics(logits, ori_mask, up_mode=True)
        self.log('test_dice', dice.item(), on_epoch=True, prog_bar=True)
        self.log('test_thres_dice', thres_dice.item(), on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):

        img = batch['image']
        dataset_name = batch['dataset_name']
        # print(f"dataset_name: {dataset_name}")

        data_name = batch['data_name']
        ori_mask = batch['ori_mask']
        logits = self.forward(img)

        return {'logits': logits, 'dataset_name':dataset_name, 'data_name':data_name, 'ori_mask': ori_mask}

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

    preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path="best", return_predictions=True)
    # print(f"preds len: {len(preds)}")

    result = {}    
    for batch in preds:
        # # save result map

        # save_dir = os.path.join(cli.trainer.log_dir, 'result_map', batch['dataset_name'][0])
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # cv2.imwrite(os.path.join(save_dir, batch['data_name'][0]), batch['pred'][0])

        logits = batch['logits']
        ori_mask = batch['ori_mask']
        dataset_name = batch['dataset_name'][0]
        data_name = batch['data_name'][0]
        
        if dataset_name not in result:
            result[dataset_name] = {}
        
        # print(f"dataset_name: {dataset_name} data_name: {data_name}")
        thres_dice = threshold_binary_dice_metrics(logits, ori_mask, up_mode=True)
        result[dataset_name][data_name] = thres_dice.item()

    result_all = {}
    for dataset in result.keys():
        result_all[dataset] =  np.mean(list(result[dataset].values()))
    pd.DataFrame.from_dict(result_all, orient='index', columns=['mDice'])\
        .to_csv(os.path.join(cli.trainer.log_dir, 'result.csv'))
        
    print(f"mean Dice: {np.mean(list(result_all.values()))}")




if __name__ == "__main__":

    cli_main()
    