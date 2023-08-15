import os
from skimage import io
import numpy as np
import torch
import torch.nn as nn
from inpaint_tools import read_file_list
from torch.utils.data import Dataset
import copy

class CatDataset(Dataset):
    def __init__(self,settings,test=False):
        input_data_dir = settings["dirs"]["input_data_dir"]
        set_type = settings["data_set"] + ".txt"

        file_list = os.path.join(input_data_dir, "data_splits", set_type)
        self.file_ids = read_file_list(file_list)
        self.test = test

        if self.test:
            self.masked_image_list = [os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png") for idx in self.file_ids]
            self.mask_list = [os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png") for idx in self.file_ids]
        else:
            self.image_list = [os.path.join(input_data_dir, "originals", f"{idx}.jpg") for idx in self.file_ids]
            self.mask_list = [os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png") for idx in self.file_ids]


    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):

        if self.test:
            im_id = self.file_ids[index]
            masked_image = io.imread(self.masked_image_list[index])
            mask = io.imread(self.mask_list[index])
            mask = torch.from_numpy(mask).unsqueeze(0).float()/255.0
            masked_image = torch.from_numpy(masked_image).permute(2,0,1).float()/255.0
            model_input = torch.cat((masked_image,mask),dim=0)
            return model_input, mask, im_id

        image = io.imread(self.image_list[index])
        mask = io.imread(np.random.choice(self.mask_list))
        masked_image = np.zeros_like(image)

        masked_image[~mask.astype(np.bool)] = image[~mask.astype(np.bool)]

        image = torch.from_numpy(image).permute(2,0,1).float()/255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()/255.0
        masked_image = torch.from_numpy(masked_image).permute(2,0,1).float()/255.0
        model_input = torch.cat((masked_image,mask),dim=0)

        return model_input, mask, image