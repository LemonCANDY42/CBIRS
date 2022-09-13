# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 13:03
# @Author  : Kenny Zhou
# @FileName: build_encode_library.py
# @Software: PyCharm
# @Email    ：l.w.r.f.42@gmail.com

import torch
from utils.tools import img2tensor
from pathlib import Path
import pytorch_ssim
from utils.inference import load_model, model_inference


class EncodeLibrary:
	"""
		Image feature encoding library
	"""
	def __init__(self, *args, image_size=(300, 300), **kwargs):
		self._image_size = image_size
		self.image_dir = None
		self.out_path = None
		self.tensor_dict = {}


	@property
	def image_size(self):
		return self._image_size

	def set_image_size(self, image_size: int or tuple):
		if isinstance(image_size, int):
			self._image_size = (image_size, image_size)
		elif isinstance(image_size, tuple):
			self._image_size = image_size
		else:
			raise Exception(f"Wrong image_size class:{type(image_size)},it's must be int or tuple")

	def make_image_batch(self, image_dir, suffix=".jpg"):
		'''
		Load all the images with the specified suffix under the folder and sort them. Make tensor batch in order.
		:param image_dir:
		:type image_dir:
		:param image_size:
		:type image_size:
		:param suffix:
		:type suffix:
		:return:
		:rtype:
		'''
		if isinstance(image_dir, str):
			self.image_dir = Path(image_dir)
		elif isinstance(image_dir, Path):
			self.image_dir = image_dir
		else:
			raise Exception(f"Wrong input class:{type(image_dir)},it's must be str or Path")

		tensor_dict = {}
		path_list = sorted(list(self.image_dir.glob(f"**/*{suffix}")), key=lambda img: int(img.stem))

		for file_name in path_list:
			tensor = img2tensor(file_name)  # .unsqueeze(0)
			tensor_dict[tensor] = file_name
		# batch = torch.cat(tensor_list,0)
		return tensor_dict

	def build_library(self, image_dir, out_path, suffix=".jpg"):
		self.out_path = out_path
		self.tensor_dict = self.make_image_batch(image_dir, suffix)
		torch.save(self.tensor_dict, self.out_path)

	def load_library(self,path):
		self.tensor_dict = torch.load(path)

	def extend_library(self,image_dir, out_path, suffix=".jpg"):
		self.out_path = out_path
		temp_tensor_dict = self.make_image_batch(image_dir, suffix)
		self.tensor_dict.update(temp_tensor_dict)
		torch.save(self.tensor_dict, self.out_path)


