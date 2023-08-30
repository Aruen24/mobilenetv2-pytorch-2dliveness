# -*- coding: utf-8 -*-
# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from skimage import feature
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import importlib
import torch
import sys
sys.path.append("./symbol")
from mobilenetv2 import MobileNetV2
from mobilenetv3 import MobileNetV3
from resnet import resnet18

import sys
importlib.reload(sys)
from PIL import Image

from torchvision import transforms




if __name__ == "__main__":
	# 读取图片进行测试
    # model = Models()
    model = resnet18()
    #model = torch.load("./liveness_detect.pth", map_location='cpu')
    model.eval().float()
    model.load_state_dict(torch.load("./models/resnet.pth", map_location='cpu'),strict=False)
    #print(model)
   # print(model)
 #   datatransform = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datatransform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for im in os.listdir('./test/'):
        im = cv2.imread(os.path.join('./test',im))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = cv2.resize(im,(112,112))
        #im = (im / 127.5) -1.0
        #input_data = torch.from_numpy(im.transpose(2, 0, 1)).float().unsqueeze(0)
        #print("===========")
        #print(input_data)

        im = Image.fromarray(im, mode='RGB')

        im = datatransform(im)
        im = im.unsqueeze(0)
        #print(im)
        #print(input_data.shape)
        #im = im.to('cpu')
        outputs = model(im)
        #outputs = model(input_data)
        _, prediction = torch.max(outputs, 1)
        print(prediction.item())
        print(outputs.data.numpy())
    #torch.save(model.state_dict(), "detect.pth")

