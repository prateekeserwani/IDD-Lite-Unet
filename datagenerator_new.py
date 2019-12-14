import numpy as np
#import cv2
#import os
#import torch
#from utils import *
#from matplotlib import pyplot as plt

import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
#import random
#import sys


class CustomDataset(Dataset):
	"""
	A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
	"""
	def __init__(self,batch_size,path,phase,max_side=1024, classes=8, visual=False):
		self.batch_size=batch_size
		self.pointer =0
		self.path = path
		self.dataset = []
		self.phase=phase
		self.load_dataset()
		self.image_size = max_side
		self.classes = classes
		self.visual=visual
		self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

	def load_dataset(self):
		print("loading data")
		train_read_path=self.path+'images/train/*/*.png'
		test_read_path=self.path+'images/test/*/*.png'
		#print(train_read_path)
		train_file_paths=glob.glob(train_read_path)
		test_file_paths=glob.glob(test_read_path)
		#print(train_file_paths)

		if not self.phase=='test':
			#parse the train_file_paths
			for paths in train_file_paths:
				image_paths=paths
				id_list=paths.split('/')[-1].split('_')[:-1]
				mask_folder=id_list[0]
				# print("id_list",id_list)
				id=''
				for el in id_list:
					id=id+str(el)+'_'
				id=id[:-1]

				mask_path='./dataset/masks/train/'+mask_folder+ '/' + id+ '_gtFine_labelIds.png'
				print(mask_path)
				self.dataset.append((paths,mask_path))
		#self.dataset.append(glob.glob(''))
		# for root, dirs, files in os.walk(os.path.join(self.path,'images',self.phase)):
		# 	#print(dirs)
		# 	for file in files:
		# 		if file.endswith(".jpg"):
		# 			folder=root.split(os.sep)[-1]
		# 			if not self.phase=='test':
		# 				self.dataset.append((os.path.join(root, file),os.path.join(self.path,'annotation',self.phase,folder, file.split('_')[0]+'_label'+'.png')))
		# 			else:
		# 				self.dataset.append((os.path.join(root, file)))
		print(self.dataset)

	def decode(self,gt):
		ground_truth = np.zeros((self.classes, self.image_size,self.image_size),dtype='uint8')
		for index in range(self.classes-1):
			ground_truth[index,0:gt.shape[-2],0:gt.shape[-1]] = (gt==index)*1
		ground_truth[self.classes-1,0:gt.shape[-2],0:gt.shape[-1]]=(gt==255)*1

		return ground_truth

	def visualize(self,image, gt):
		plt.imshow(image)
		plt.show()
		for index in range(self.classes):
			plt.imshow(gt[index,...])
			plt.show()

	def read_image_and_gt(self,data):
		file_name=data[0]

		image = cv2.imread(file_name)
		h,w,c = image.shape
		scale=h
		if w>h:
			scale=w
		scale=self.image_size/scale
		image = cv2.resize(image,(min(int(w*scale),self.image_size),min(int(h*scale),self.image_size)), interpolation=cv2.INTER_CUBIC)

		new_image = np.zeros((self.image_size,self.image_size,3 ),dtype='uint8')
		new_image[0:image.shape[-3],0:image.shape[-2],:]=image.copy()

		if not self.phase =='test':
			gt_name=data[1]
			gt = cv2.imread(gt_name)
			#print(gt_name)
			#print( 'gt shape', gt.shape)
			gt = cv2.resize(gt,(min(int(w*scale),self.image_size),min(int(h*scale),self.image_size)), interpolation=cv2.INTER_NEAREST)
			gt = self.decode(gt[:,:,0])
			if self.visual:
				self.visualize(new_image, gt)
			gt = gt.astype('float32')
			#new_image = new_image.transpose(2,0,1)
			new_image = self.transform(new_image)
			return new_image, gt
		else:
			#print('testing --------------------')
			new_image = self.transform(new_image)
			return new_image


	def __getitem__(self, index):
		# Remember, the Nth caption corresponds to the (N // captions_per_image)th images
		image_batch = torch.zeros(3,self.image_size,2*self.image_size).type(torch.FloatTensor)
		gt_batch = torch.zeros(self.classes,self.image_size,2*self.image_size).type(torch.uint8)

		if self.phase=='test':
			image_name = self.dataset[index]
			#print('testing --------------------')
			image = self.read_image_and_gt([image_name])
			return {'image': image, 'file_name':image_name}
		else:
			image_name, gt_name = self.dataset[index]
			#print('hello------------------------')
			image, gt = self.read_image_and_gt([image_name, gt_name] )
			return {'image': image, 'gt':gt, 'file_name':image_name}

	def __len__(self):
		return len(self.dataset)
