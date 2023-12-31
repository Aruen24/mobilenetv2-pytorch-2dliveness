import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import mobileNetv2
import numpy as np
import cv2
import copy
import sys
from dataset import Images, get_liveness_data


SEED = 42
torch.manual_seed(SEED)

# train_dataset = Images(path="/home/wyw/train.txt")
# train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
# val_dataset = Images(path="/home/wyw/valid.txt")
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

train_dataset = get_liveness_data('/home/disk04/wyw_data/irDatas/irTrain0719', 'train',True)
val_dataset = get_liveness_data('/home/disk04/wyw_data/irDatas/irTest0719', 'test',True)

# train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, shuffle=False, drop_last=False)

model = mobileNetv2().cuda()
device = torch.device('cuda')

model = nn.DataParallel(model)
model.to(device)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

model.apply(weight_init)

num_epochs = 42
#num_epochs = 160
learning_rate = 1e-4
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

print("Training...")
best_acc = 0
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.float()
        train_images = train_images.cuda()
        train_labels = train_labels.view(-1, 1).cuda()
        one_hot = torch.zeros(train_labels.shape[0], 2).cuda().scatter_(1, train_labels, 1)
        optimizer.zero_grad()
        output = model(train_images)
        loss = criterion(output, one_hot)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        if i%50==0 and i > 0:
            print("Epoch: " + str(epoch+1) + "| train loss: " + str(total_train_loss/i))

    print("Epoch: " + str(epoch+1) + "| train loss: " + str(total_train_loss/len(train_loader)))

    model.eval()
    with torch.no_grad():
        correct = 0.0
        for val_image, val_label in val_loader:
            val_image = val_image.float()
            val_image = val_image.cuda()
            val_label = val_label.cuda()
            val_output = model(val_image)
            _, prediction = torch.max(val_output, 1)
            if val_label == prediction:
                correct += 1

        print("Epoch {}: Val acc is {}%".format(epoch+1, correct*100/len(val_loader)))
        epoch_acc = correct / len(val_loader)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, './weights/epoch_' + str((epoch+1)) + '.pth')
