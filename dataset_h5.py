from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

import torchvision.transforms as transforms # 画像拡張
from generator import Generator # 生成器
import torchvision               # PIL変換用

from PIL import Image
import h5py

from random import randrange

import torch.multiprocessing as multiprocessing # cycleganとclamの両方でAIを使うため


'''
CycleGAN推論クラス
'''
class cycleGan():
    def __init__(self):
        self.batch_size = 1  # バッチサイズは1に固定する
        self.size = 256 # パッチのサイズ
        self.input_nc = 3 # 入力チャンネル数
        self.output_nc = 3# 出力チャンネル数
        self.generator_A2B = './netG_A2B.pth'   # AからBへ変換するモデルのパス
        self.cuda = True
        self.image_save = True                        # cyclegan実施前後のパッチ画像を保存するかどうか(チェック用)
        self.save_num   = 10                          # cyclegan実施前後のパッチ画像を保存する枚数
        self.out_dir = './cgan_out'                   #  cyclegan実施前後のパッチ画像の保存先ディレクトリ名
                                                      # (実施前のディレクトリ名は+"_pre",実施後は+"_post"となる)
    # CycleGAN推論メソッド
    def inference(self, img, idx):
        #print("item_A_o:",item_A_o.size)
        img = img.convert("RGB")
        if self.image_save and idx <= self.save_num:
            # cyclegan実施前の画像保存先ディレクトリ
            out_pre = self.out_dir + '_pre'
            os.makedirs(out_pre, exist_ok=True)
            # 番号をつけて
            img.save(os.path.join(out_pre, f"image_A_to_B_pre_{idx}.png"))

        # 拡張パラメータ
        transforms_ = [transforms.Resize(int(self.size*1.0), Image.BICUBIC), 
                       transforms.RandomCrop(self.size), 
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        transform = transforms.Compose(transforms_)
        
        # テンソル変換、標準化など実施
        img = transform(img)

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        input_img = Tensor(self.batch_size, self.input_nc, self.size, self.size)  #貼り付ける元
#        print("input_imgのサイズ:",input_img.size())

        # 生成器の用意
        netG_A2B = Generator(self.input_nc, self.output_nc)

        # 生成器をcudaに対応
        if self.cuda:
            netG_A2B.cuda()

        # モデルの読み込み
        netG_A2B.load_state_dict(torch.load(self.generator_A2B))
        #fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        # 変換
        fake_img = 0.5*(netG_A2B(input_img.copy_(img)).data + 1.0)

        ## 画像を白い元画像に貼り付ける
        # バッチ数1でないとエラーになるが、バッチ次元を削減
        out_img = fake_img.view(fake_img.size()[1],fake_img.size()[2],fake_img.size()[3])
        # PIL形式に変換(これを返す)
        out_img_PIL = torchvision.transforms.functional.to_pil_image(out_img)
#        print("サイズ:",out_img_PIL.size)
        if self.image_save and idx <= self.save_num:
            out_post = self.out_dir + '_post'
            os.makedirs(out_post, exist_ok=True)
            out_img_PIL.save(os.path.join(out_post, f"image_A_to_B_post_{idx}.png"))
        
        # cpuに戻す
        return out_img_PIL

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
# 23.7.30 cyclegan推論導入用, 引数cycle_gan追加
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1,
		cycle_gan=False
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		# 23.7.30 cyclegan推論導入用, 引数追加
		self.cycle_gan = cycle_gan
		# 以上、23.7.30 cyclegan推論導入用, 引数追加
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			'''
			細胞特徴抽出のためのアレンジ③
			'''
			print(f"image_dimensions:{self.wsi.level_dimensions[self.patch_level]}")
			# 画像の大きさを確認し、
			# もしもwかhが32,768を超えるとエラーが出るため、seg_levelを0=>1に変更する。
			if self.patch_level==0 and (self.wsi.level_dimensions[self.patch_level][0] >= 32768 or self.wsi.level_dimensions[self.patch_level][1] >= 32768):
				print("self.patch_level is set from 0 to 1")
				self.patch_level = 1
			'''
			細胞特徴抽出のためのアレンジ③終了
			'''
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
#		print("読み込み画像タイプ:",type(img))      
		# 23.7.30cyclegan推論導入用
		# cycle_gan使用の場合
		if self.cycle_gan:
			# cycleGanクラスのインスタンスを作成
			cgan = cycleGan()
			# cycleGanを実施し、画像を変換
			img  = cgan.inference(img, idx)         
#			print("変換後画像タイプ:",type(img))      
            
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




