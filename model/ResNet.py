# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 0:16
# @Author  : Kenny Zhou
# @FileName: ResNet.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet50,ResNet50_Weights
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup


class ResNet50(nn.Module):
  def __init__(self):
    super().__init__()
    self.modle = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True)
    self.encoder = nn.Sequential(
      nn.Linear(1000, 4096),
      # nn.ReLU(),
      # nn.Linear(512, 10)
    )
    for param in self.modle.parameters():
      param.requires_grad = False

  def forward(self, imgs):
    x = self.encoder(self.modle(imgs))
    y = x.view(x.size(0),1,64,64)
    return y