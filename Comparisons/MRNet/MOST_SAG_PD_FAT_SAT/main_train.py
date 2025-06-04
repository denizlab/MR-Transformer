import argparse
import datetime
import numpy as np
import pandas as pd
import os
import time
import copy
import random
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import MRNet
from augmentation import *



if torch.cuda.is_available:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    
def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    return None
print(device)
setup_seed(0)




parser = argparse.ArgumentParser('MRNet', add_help=False)
parser.add_argument('--train_csv', type=str)
parser.add_argument('--val_csv', type=str)
parser.add_argument('--mri_path', type=str)
parser.add_argument('--save_name', type=str)
args = parser.parse_args()

train_csv = args.train_csv
val_csv = args.val_csv
mri_path = args.mri_path
save_name = args.save_name

batch_size = 1
num_worker = 8
channels = 3
image_size = [36,512,512]


class MRI_dataset(Dataset):
    def __init__(self,
                 data_df_path,
                 train = True):
        
        self.data_df = pd.read_csv(data_df_path)
        self.train = train
        self.label_index = 'Label'
        self.image_path_index = 'FileName'
        self.t2map_images = False
        self.tse_images = True
        self.image_size = image_size
        self.num_MR_slices = image_size[0]
        self.channels = 3
        self.aug_RandomRescale = True
        self.aug_resize_scale = (0.9,1.3)
        self.aug_degrees = 10
        self.aug_RandomFlip = True
        self.aug_GaussianBlur = True
        
        
    def __getitem__(self, index):

        image_path = os.path.join(mri_path,self.data_df.iloc[index][self.image_path_index])
        
        if self.t2map_images:
            image = read_t2mapping_hdf5(image_path, self.num_MR_slices)
        elif self.tse_images:
            image = read_iw_tse_hdf5(image_path, self.num_MR_slices)
            
        if self.train:
            if self.aug_RandomRescale:
                image = RandomRescale2D(self.aug_resize_scale)(image)
            image = SimpleRotate(self.aug_degrees)(image)
            image = RandomCrop(self.image_size)(image)
            image = Standardize()(image)
            if self.aug_RandomFlip:
                image = RandomFlip()(image)
            if self.aug_GaussianBlur:
                image = GaussianBlur()(image)  
            image = ToTensor()(image)
            image = RepeatChannels(self.channels)(image)
            
        else: 
            image = CenterCrop(self.image_size)(image)
            image = Standardize()(image)
            image = ToTensor()(image)
            image = RepeatChannels(self.channels)(image)
            
        label = self.data_df.iloc[index][self.label_index]
        label = torch.tensor(label, dtype=torch.long)
        
        return image.float(), label, index
    
    def __len__(self):
        return len(self.data_df)


    
train_loader = DataLoader(MRI_dataset(data_df_path=train_csv, train=True),\
                          batch_size=batch_size, num_workers=num_worker, pin_memory=True, shuffle=True)
val_loader = DataLoader(MRI_dataset(data_df_path=val_csv, train=False),\
                          batch_size=batch_size, num_workers=num_worker, pin_memory=True, shuffle=False)

    
    
    
def train_fun(model, train_loader=train_loader, val_loader=val_loader, num_epoch=50,file_name=save_name):
    
    start_time = time.time()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)

    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_auc = -1
    
    for epoch in range(num_epoch):
        
        # Training steps
        pred_epoch = []
        pred_scores_epoch = []
        truths_epoch = []
        loss_epoch = []
        
        model.train()
        for i, (sample) in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample[0].to(device)
            labels = sample[1].to(device)
            image = image.transpose(1,2)
            outputs = model(image)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            pred_score = nn.Softmax(1)(outputs).cpu().detach().numpy()
            pred_scores_epoch += pred_score[:,1].tolist()
            pred  = np.argmax(pred_score,axis=1)
            pred_epoch += pred.tolist()
            truths_epoch += labels.cpu().numpy().tolist()
            
        correct_num = (np.array(pred_epoch) == np.array(truths_epoch)).sum()
        total_num = len(truths_epoch)
        acc = correct_num/total_num
        auc = roc_auc_score(np.array(truths_epoch),np.array(pred_scores_epoch))
        
        with open(file_name+'.txt','a') as file0:
            print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,num_epoch),file=file0)
            print('Train set | Loss: {:6.4f} | Accuracy: {:4.2f}% | AUC: {:6.4f}'\
                  .format(np.mean(loss_epoch), acc*100, auc),file=file0)
        
        model.eval()
                  
        pred_epoch = []
        pred_scores_epoch = []
        truths_epoch = []
        loss_epoch = []
        
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                image = sample[0].to(device)
                labels = sample[1].to(device)
                image = image.transpose(1,2)
                outputs = model(image)
                loss = loss_fn(outputs, labels)

                loss_epoch.append(loss.item())
                pred_score = nn.Softmax(1)(outputs).cpu().detach().numpy()
                pred_scores_epoch += pred_score[:,1].tolist()
                pred  = np.argmax(pred_score,axis=1)
                pred_epoch += pred.tolist()
                truths_epoch += labels.cpu().numpy().tolist()
                
        correct_num = (np.array(pred_epoch) == np.array(truths_epoch)).sum()
        total_num = len(truths_epoch)
        acc = correct_num/total_num
        auc = roc_auc_score(np.array(truths_epoch),np.array(pred_scores_epoch))
        avg_loss = np.mean(loss_epoch)
        scheduler.step(avg_loss)

        if auc > best_auc:
            best_auc = auc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        save_state = {'best_model_wts': best_model_wts,
                      'model': model.state_dict(),
                      'optimizer': optimizer,
                      'scheduler': scheduler}
        
        torch.save(save_state, file_name+'.pt')
        
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        with open(file_name+'.txt','a') as file0:
            print('Val set   | Loss: {:6.4f} | Accuracy: {:4.2f}% | AUC: {:6.4f} | Best AUC: {:6.4f} | time elapse: {:>9}'\
                      .format(np.mean(loss_epoch), acc*100, auc, best_auc, elapse),file=file0)
                
    return None


model = MRNet().to(device)
train_fun(model)
