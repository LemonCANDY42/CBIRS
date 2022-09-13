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


def load_model(model_path):
	net = LitAutoEncoder.load_from_checkpoint(model_path)
	# torch.set_grad_enabled(False)
	net.eval()
	return net

def model_inference(image_tensor,net):
	x = image_tensor.unsqueeze(0)
	with torch.no_grad():
		out = net(x)
	return out


if __name__ =="__main__":
	ssim_loss = pytorch_ssim.SSIM(window_size=64)

	model_path = Path("./lightning_logs/version_3/checkpoints/epoch=224-step=1350.ckpt")
	net = load_model(model_path)

	image_path_0 = Path("/Volumes/Sandi/Jewelry/R/CRE6981R01M14WA.jpg")
	image_tensor_0 = img2tensor(image_path_0)
	result0 = model_inference(image_tensor_0,net)

	image_path_1 = Path("/Volumes/Sandi/Jewelry/R/CRG5747R02WM18RA.jpg")
	image_tensor_1 = img2tensor(image_path_1)
	result1 = model_inference(image_tensor_1,net)

	loss = ssim_loss(result0,result1)
	print(loss.item())