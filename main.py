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
from utils import binary_dice_metric

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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        img, seg = batch['image'], batch['label']
        logits = self.forward(img)
        loss = self.loss(logits, seg)
        dice = binary_dice_metric(logits, seg)

        self.log('val_loss', loss.item(), prog_bar=True)
        self.log('dice', dice.item(), prog_bar=True)


    def test_step(self, batch, batch_idx):
        img, seg = batch['image'], batch['label']
        logits = self.forward(img)
        dice = binary_dice_metric(logits, seg)
        self.log('test_dice', dice.item(), prog_bar=True)

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


if __name__ == "__main__":

    cli_main()
    