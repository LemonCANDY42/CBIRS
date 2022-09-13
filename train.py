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
from model.base_model import LitAutoEncoder

from torchvision.transforms import InterpolationMode
from pytorch_lightning.callbacks import ModelCheckpoint


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

checkpoint_callback = ModelCheckpoint(monitor="train_loss")


# training
# trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=1, precision=32, limit_train_batches=0.5,callbacks=[checkpoint_callback],min_epochs=100,max_epochs=1000)

#断点续训
trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=1, precision=32, limit_train_batches=0.5,min_epochs=100,max_epochs=1000,callbacks=[checkpoint_callback],resume_from_checkpoint=r'./lightning_logs/version_0/checkpoints/epoch=78-step=474.ckpt')

trainer.fit(model, train_loader)

