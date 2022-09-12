# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 0:08
# @Author  : Kenny Zhou
# @FileName: train.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.datasets
from torchvision import transforms
import pytorch_lightning as pl
from model.ResNet import ResNet50
from utils.augmentation import tfs_img
from torchvision.transforms import InterpolationMode
import pytorch_ssim


class LitAutoEncoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.encoder = ResNet50()
    self.ssim_loss = pytorch_ssim.SSIM(window_size=32)

  def forward(self, x):
    embedding = self.encoder(x)
    return embedding

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def training_step(self, train_batch, batch_idx):
    x,_ = train_batch
    tfs = tfs_img((300,300))
    x = tfs(x)
    y = tfs(x)
    # x = x.view(x.size(0), -1)
    # y = y.view(y.size(0), -1)
    print(x.shape)
    x = self.encoder(x)
    y = self.encoder(y)

    loss = self.ssim_loss(x, y)
    # loss = F.mse_loss(x_hat, x)
    self.log('train_loss', loss)
    return loss

  # def validation_step(self, val_batch, batch_idx):
  #   pass
    # x, y = val_batch
    # x = x.view(x.size(0), -1)
    # z = self.encoder(x)
    # x_hat = self.decoder(z)
    # loss = F.mse_loss(x_hat, x)
    # self.log('val_loss', loss)


# data
# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
data_dir = r"E:\Dataset\Jewelry"

tfs = transforms.Compose([
  transforms.Resize(size=(300,300), interpolation=InterpolationMode.BILINEAR),
  transforms.ToTensor(),
])

trainset = torchvision.datasets.ImageFolder(data_dir, transform=tfs)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
# val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader)

