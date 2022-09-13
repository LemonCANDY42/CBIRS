# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 11:27
# @Author  : Kenny Zhou
# @FileName: tools.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .augmentation import tfs_resize

CONVERT_TENSOR = transforms.ToTensor()

def img2tensor(image_path,image_size=(300,300)):
	img = Image.open(image_path)
	tfs = tfs_resize(image_size)
	img = tfs(img)
	return CONVERT_TENSOR(img)

if __name__ =="__main__":
	image_path = "/Volumes/Sandi/Jewelry/R/DRH7856R01MH18WA.jpg"
	tensor = img2tensor(image_path)
	print(tensor,tensor.unsqueeze(0).shape)