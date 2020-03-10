import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class Data_Generator(Dataset):
    def __init__(self, data, base_dir, transformer=None):
        self.data = data
        self.base_dir = base_dir
        self.transformer = transformer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_label = self.base_dir + self.data["id"][idx] + ".tif"
        img_class = self.data["label"][idx].reshape(1)
        img_class = img_class.astype(np.int64)
        img = Image.open(img_label)
        if self.transformer == None:
            img = np.array(img).transpose(2, 0, 1)/255
            img = img.astype(np.float32)
            return torch.from_numpy(img), torch.from_numpy(img_class)
        else:
            img = self.transformer(img)
            return img, torch.from_numpy(img_class)

def generate_train_validation_dataloader(data_train, data_val, batch_size, base_dir):
    train_loader = DataLoader(Data_Generator(data_train, base_dir), batch_size)
    validation_loader = DataLoader(Data_Generator(data_val, base_dir), batch_size)
    return train_loader, validation_loader

if __name__ == "__main__":
    pass