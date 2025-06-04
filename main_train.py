import argparse
import numpy as np
import os
import datetime
import time
from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from model import MR_Transformer
from util.dataloader import MRI_dataset



parser = argparse.ArgumentParser()

# Metadata
parser.add_argument('--train_df_path', default='Data/train.csv', type=str)
parser.add_argument('--val_df_path', default='Data/val.csv', type=str)
parser.add_argument('--mri_path', default='Data/MRI/', type=str)
parser.add_argument('--label_index', default='Label', type=str)
parser.add_argument('--image_path_index', default='FileName', type=str)
parser.add_argument('--output_file', default='output.txt', type=str)
parser.add_argument('--save_dir', default='.', type=str)
parser.add_argument('--save_model_name', default='MR_Transformer', type=str)

# MRI
parser.add_argument('--num_mr_slice', default=36, type=int)
parser.add_argument('--mr_slice_size', default=384, type=int)
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--t2map_images', default=False, type=bool)
parser.add_argument('--tse_images', default=True, type=bool)

# Data Augmentation
parser.add_argument('--aug_RandomRescale', default=True, type=bool)
parser.add_argument('--aug_resize_scale_min', default=0.9, type=float)
parser.add_argument('--aug_resize_scale_max', default=1.3, type=float)
parser.add_argument('--aug_degrees', default=10)
parser.add_argument('--aug_RandomFlip', default=True, type=bool)
parser.add_argument('--aug_GaussianBlur', default=True, type=bool)

# Model
parser.add_argument('--base_model', default='tiny', choices=['tiny', 'small', 'base'])
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--use_checkpoint', default=False, type=bool)

# Training
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_worker', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'])
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--sgd_momentum', default=0.9, type=float)
parser.add_argument('--adam_betas', default=(0.9, 0.999))
parser.add_argument('--adamw_betas', default=(0.9, 0.999))

args = parser.parse_args()


if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    return None


def main(args):
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    setup_seed(args.seed)
    
    train_loader = DataLoader(MRI_dataset(args, args.train_df_path, args.mri_path, train=True),
                              batch_size=args.batch_size,
                              num_workers=args.num_worker,
                              pin_memory=True,
                              shuffle=True)
    
    val_loader = DataLoader(MRI_dataset(args, args.val_df_path, args.mri_path, train=False),
                            batch_size=args.batch_size,
                            num_workers=args.num_worker,
                            pin_memory=True,
                            shuffle=False)
    
    model = MR_Transformer(base_model = args.base_model,
                           num_mr_slice = args.num_mr_slice,
                           mr_slice_size = args.mr_slice_size,
                           num_classes = args.num_classes,
                           use_checkpoint = args.use_checkpoint)
    
    model.to(device)
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.lr, 
                                    momentum=args.sgd_momentum, 
                                    weight_decay=args.weight_decay)
        
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, 
                                     betas=args.adam_betas, 
                                     weight_decay=args.weight_decay)
        
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, 
                                     betas=args.adamw_betas, 
                                     weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_auc = 0
    
    start_time = time.time()
    for epoch in range(args.epochs):
        
        model.train()
        model.use_checkpoint = args.use_checkpoint
        
        pred_epoch = []
        pred_scores_epoch = []
        truths_epoch = []
        loss_epoch = []
        
        for i, (sample) in enumerate(train_loader):
            optimizer.zero_grad()
            images = sample[0].to(device)
            labels = sample[1].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_epoch.append(loss.item())
            pred_score = nn.Softmax(1)(outputs).cpu().detach().numpy()
            pred_scores_epoch += pred_score[:,1].tolist()
            pred  = np.argmax(pred_score,axis=1)
            pred_epoch += pred.tolist()
            truths_epoch += labels.cpu().numpy().tolist()
            
        scheduler.step()
        
        correct_num = (np.array(pred_epoch) == np.array(truths_epoch)).sum()
        total_num = len(truths_epoch)
        acc = correct_num/total_num
        auc = roc_auc_score(np.array(truths_epoch),np.array(pred_scores_epoch))
        
        with open(os.path.join(args.save_dir, args.output_file), 'a') as file0:
            print('----------Epoch{:2d}/{:2d}----------'.format(epoch+1,args.epochs),file=file0)
            print('Train set | Loss: {:6.4f} | Accuracy: {:4.2f}% | AUC: {:6.4f}'.format(np.mean(loss_epoch), acc*100, auc),file=file0)
            

        model.eval()
        model.use_checkpoint = False
                  
        pred_epoch = []
        pred_scores_epoch = []
        truths_epoch = []
        loss_epoch = []
                  
        with torch.no_grad():
            for i, (sample) in enumerate(val_loader):
                images = sample[0].to(device)
                labels = sample[1].to(device)
                outputs = model(images)
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
        
        if auc > best_auc:
            best_auc = auc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        save_state = {'best_model_wts': best_model_wts,
                      'model': model.state_dict(),
                      'optimizer': optimizer,
                      'scheduler': scheduler}
        torch.save(save_state, os.path.join(args.save_dir, args.save_model_name + '.pt'))
            
        elapse = time.strftime('%H:%M:%S', time.gmtime(int((time.time() - start_time))))
        elapse_day = int((time.time() - start_time)//(24 * 3600))
        with open(os.path.join(args.save_dir, args.output_file), 'a') as file0:
            print('Val set   | Loss: {:6.4f} | Accuracy: {:4.2f}% | AUC: {:6.4f} | Best AUC: {:6.4f} | time elapse: {:d} day,{:>9}'.format(np.mean(loss_epoch), acc*100, auc, best_auc, elapse_day, elapse),file=file0)

            
if __name__ == '__main__':   
    main(args)
    