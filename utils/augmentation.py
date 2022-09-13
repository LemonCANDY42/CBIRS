# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 0:31
# @Author  : Kenny Zhou
# @FileName: augmentation.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

def tfs_resize(img_size):
  tfs = transforms.Compose([
    transforms.Resize(size=img_size, interpolation=InterpolationMode.BILINEAR),
  ])
  return tfs

def tfs_img(image_size):

  tfs = transforms.Compose([
    transforms.Resize(size=image_size, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((0,90)),
    transforms.RandomAffine((0, 30)),
    transforms.RandomGrayscale(p=0.01),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.1, fill=0),
    transforms.GaussianBlur(kernel_size=(5,5),
                                        sigma=(0.1, 10.0)),
    # transforms.ToTensor(),
  ])
  return tfs