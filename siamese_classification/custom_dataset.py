import os
import pandas as pd
import PIL.Image as Image
import numpy as np
import torch as th
from torch.utils.data import Dataset

DATASET1_PATH = "./datasets/OLI/OLI"
DATASET2_PATH = "./datasets/OLIVINE/OLIVINE"

class SiameseDataset(Dataset):
    def __init__(self,training_csv=None,train_dir1=None,train_dir2=None,transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv("./datasets/dataset.csv")
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir1 = train_dir1
        self.train_dir2 = train_dir2
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        if self.train_df.iat[index,0].split('_')[0] == "OLI":
            image1_path = os.path.join(self.train_dir1,self.train_df.iat[index,0])
        else:
            image1_path = os.path.join(self.train_dir2, self.train_df.iat[index, 0])

        if self.train_df.iat[index,1].split('_')[0] == "OLIVINE":
            image2_path = os.path.join(self.train_dir2, self.train_df.iat[index,1])
        else:
            image2_path = os.path.join(self.train_dir1, self.train_df.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, th.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))

    def __len__(self):
        return len(self.train_df)