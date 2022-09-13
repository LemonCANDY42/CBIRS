# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 13:05
# @Author  : Kenny Zhou
# @FileName: inference.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

import torch
import pytorch_lightning as pl
from model.base_model import LitAutoEncoder
from utils.tools import img2tensor
from pathlib import Path

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