import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class CTReportDataset(Dataset):
    def __init__(self, args, data_folder, xlsx_file, tokenizer,long_file,  min_slices=20, resize_dim=500, num_frames=2, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.accession_to_text = self.load_accession_text(xlsx_file)
        self.paths=[]
        self.long_file = long_file
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def load_accession_text(self, xlsx_file):
        df = pd.read_excel(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['AccessionNo']] = row["Findings_EN"]
        return accession_to_text


    def prepare_samples(self):
        samples = []
        df = pd.read_csv(self.long_file)
        for i, col in df.iterrows():
            old_img = col["OlderVolume"]
            new_img = col["NewerVolume"]
            accession_old = old_img.split("/")[1]
            accession_new = new_img.split("/")[1]

            text_old = self.accession_to_text[accession_old]
            text_new = self.accession_to_text[accession_new]

            sample = (old_img, new_img, text_old, text_new)
            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        paths = path.split(".")
        path = paths[0]+".npz"
        path_img = self.data_folder+path
        img_data = np.load(path_img)['arr_0']

        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        #for i in range(img_data.shape[2]):
        #    img_slice = Image.fromarray(img_data[:, :, i], mode='F')
        #    img_transformed = transform(img_slice)
        #    slices.append(img_transformed)
        #tensor = torch.stack(slices,dim=1)
        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)
        
        tensor = tensor.unsqueeze(0)
        return tensor

    
    def __getitem__(self, index):
        old_img, new_img, text_old, text_new = self.samples[index]

        old_tensor = self.nii_to_tensor(old_img)
        new_tensor = self.nii_to_tensor(new_img)
        ids_old = self.tokenizer(text_old)[:self.max_seq_length]
        mask_old = [1] * len(ids_old)
        ids_new = self.tokenizer(text_new)[:self.max_seq_length]
        mask_new = [1] * len(ids_new)
        seq_lenght = len(ids_new)
        context_lenght = len(ids_old)
        sample = (new_img, new_tensor, ids_new, mask_new, seq_lenght, ids_old, mask_old, context_lenght, old_tensor)
        return sample
