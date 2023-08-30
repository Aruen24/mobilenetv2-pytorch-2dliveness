import torch
import torch.nn as nn
import numpy as np
from model import mobileNetv2
import cv2
import os
import sys

#model = mobileNetv2().cuda()
model = mobileNetv2()
#model.load_state_dict(torch.load("/home/wangyuanwen/mobilenetv2-pytorch/weights/epoch_1.pth"))

#model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./weights/epoch_25.pth').items()})

# dist = torch.load('./weights/epoch_25.pth',map_location='cpu')
# nek = {}
# for k, v in dist.items():
#     nek[k[7: ]] = v
# model.load_state_dict(nek)
# torch.save(model.state_dict(), './model_new.pth',_use_new_zipfile_serialization=False)

model.load_state_dict(torch.load("./models/model_new.pth"))
root = "./data/mask"
img_list = os.listdir(root)



#0-mask   1-nomask
model.eval()
pred_pos = 0
with torch.no_grad():
    for img_name in img_list:
        if ".jpg" in img_name:
            img = cv2.imread(os.path.join(root, img_name))
            print(img)
            img = cv2.resize(img, (112, 112)).T
            print(img.shape)
            print(img)
            img = img / 255
            img = torch.from_numpy(np.expand_dims(img, axis=0))
            #img = img.float().cuda()
            img = img.float()

            output = model(img)
            _, pred = torch.max(output, 1)
            #print(output.cpu().float())
            if int(pred) == 1:
                pred_pos += 1

print(str(pred_pos) + "/" + str(len(img_list)))
