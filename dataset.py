import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as trans




class Images(Dataset):
    def __init__(self, path):
        f = open(path, "r")
        self.imgs = []
        for line in f.readlines():
            img_path = line.strip("\n").split(" ")[0]
            img_label = line.strip("\n").split(" ")[1]
            self.imgs.append((img_path, img_label))
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        tup = self.imgs[index]
        img = cv2.imread(tup[0],-1)

        if len(img.shape) == 2:
            img = img.reshape((1, 112, 96))
        else:
            img = img.transpose((2,0,1))
            img = img[1, :, :].astype(np.float32)
            img = img.reshape((1, 112, 96))
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        label = int(tup[1])
        return img, label


class get_liveness_data(Dataset):
    def __init__(self,data_path,flag,isnorm):
        self.data=self.make_data(data_path)
        if flag=='test':
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])
        else:
            if isnorm:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    trans.Normalize([0.5], [0.5]),
                ])
            else:
                self.trans = trans.Compose([
                    trans.Grayscale(num_output_channels=1),
                    trans.RandomResizedCrop((224, 224), scale=(0.92, 1), ratio=(1, 1)),
                    trans.RandomHorizontalFlip(),
                    trans.ColorJitter(brightness=0.5),
                    trans.ToTensor(),
                    # trans.Normalize([0.5], [0.5]),
                ])


    def __getitem__(self, index):
        img_path=self.data[index][0]
        label = self.data[index][1]
        img=Image.open(img_path)
        img=self.trans(img)
        img = 255 * img
        return img, label

    def make_data(self,data_path):
        data_lines=[]
        negs=0
        posts=0
        for path,dirs,files in os.walk(data_path):
            imgs = list(filter(lambda x: x.endswith(('png','jpg')), files))
            if imgs!=[]:
                img_paths=[os.path.join(path,img)for img in imgs]
                for img_path in img_paths:
                    label=0
                    if 'real' in img_path:
                        label=1
                        posts+=1
                    elif 'fake' in img_path:
                        label=0
                        negs+=1
                    data_lines.append((img_path,label))
        print('total sample is %d, postive sample is %d, negative sample is %d'%((posts+negs),posts,negs))
        return data_lines
    def __len__(self):
        return len(self.data)
