import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from archs import NestedUNet, UNet
from msdataloader import MedicalSegDataModule
from pytorch_lightning.cli import LightningCLI
from torch import optim
from torch.optim.lr_scheduler import StepLR
from utils import binary_dice_metric, threshold_binary_dice_metrics
import cv2
import os

class SegModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.model_name = model_name
        self.net = eval(model_name)(num_classes=1, input_channels=3)
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        return self.net(x)
    

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)
    

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
        data_name = batch['data_name']
        ori_mask = batch['ori_mask']
        logits = self.forward(img)

        pred = F.interpolate(logits, ori_mask.shape[-2:], mode='bilinear', align_corners=True)
        pred = pred.sigmoid().data.cpu().squeeze().numpy()
        pred = pred - pred.min() / (pred.max() - pred.min() + 1e-8)
        pred = (pred*255).astype(np.uint8)

        return {'pred':pred, 'dataset_name':dataset_name, 'data_name':data_name}

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
        

class SegModelNestedUNet(SegModel):
    def __init__(self, learning_rate,):
        super().__init__(learning_rate)

        self.net = NestedUNet(num_classes=1, input_channels=3)


def cli_main():
    cli = LightningCLI(datamodule_class=MedicalSegDataModule, model_class=SegModel, seed_everything_default=False, run=False)  
   
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")

    # preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path="best", return_predictions=True)
    
    # # save result map
    # for batch in preds:
    #     save_dir = os.path.join(cli.trainer.log_dir, 'result_map', batch['dataset_name'][0])
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     cv2.imwrite(os.path.join(save_dir, batch['data_name'][0]), batch['pred'][0])


if __name__ == "__main__":

    cli_main()
    