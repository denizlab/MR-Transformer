import pandas as pd
import os
from torch.utils.data import Dataset
from .augmentation import *


class MRI_dataset(Dataset):
    def __init__(self,
                 args,
                 data_df_path,
                 mri_path,
                 train = True):
        
        self.data_df = pd.read_csv(data_df_path)
        self.mri_path = mri_path
        self.train = train
        self.label_index = args.label_index # 'Label'
        self.image_path_index = args.image_path_index # 'FileName'
        self.t2map_images = args.t2map_images # False
        self.tse_images = args.tse_images # True
        self.image_size = [args.num_mr_slice, args.mr_slice_size, args.mr_slice_size] # [36, 384, 384]
        self.num_MR_slices = args.num_mr_slice # 36
        self.channels = args.channels # 3
        self.aug_RandomRescale = args.aug_RandomRescale # True
        self.aug_resize_scale = (args.aug_resize_scale_min, args.aug_resize_scale_max) # (0.9,1.3)
        self.aug_degrees = args.aug_degrees # 10
        self.aug_RandomFlip = args.aug_RandomFlip # True
        self.aug_GaussianBlur = args.aug_GaussianBlur # True
        
        
    def __getitem__(self, index):

        image_path = os.path.join(self.mri_path,self.data_df.iloc[index][self.image_path_index])
        
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
