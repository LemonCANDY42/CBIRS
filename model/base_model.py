# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 11:14
# @Author  : Kenny Zhou
# @FileName: base_model.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from model.ResNet import ResNet50
from utils.augmentation import tfs_img
import pytorch_ssim

class LitAutoEncoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.encoder = ResNet50()
    self.ssim_loss = pytorch_ssim.SSIM(window_size=64)

  def forward(self, x):
    embedding = self.encoder(x)
    return embedding

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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