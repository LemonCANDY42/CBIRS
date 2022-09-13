# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 11:16
# @Author  : Kenny Zhou
# @FileName: test.py
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

from utils.tools import img2tensor
from pathlib import Path
import pytorch_ssim

from utils.inference import load_model,model_inference


if __name__ =="__main__":
	ssim_loss = pytorch_ssim.SSIM(window_size=64)

	model_path = Path("./lightning_logs/version_3/checkpoints/epoch=224-step=1350.ckpt")
	net = load_model(model_path)

	image_path_0 = Path("/Volumes/Sandi/Jewelry/R/DRD1523R01WM18WA.jpg")
	image_tensor_0 = img2tensor(image_path_0)
	result0 = model_inference(image_tensor_0,net)

	image_path_1 = Path("/Volumes/Sandi/Jewelry/R/DRD1521R01WM18YA.jpg")
	image_tensor_1 = img2tensor(image_path_1)
	result1 = model_inference(image_tensor_1,net)

	loss = ssim_loss(result0,result1)
	print(loss.item())