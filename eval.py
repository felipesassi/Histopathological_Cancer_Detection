import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.model import TL_ResNet50
from datasets.datasets import generate_train_validation_dataloader
from models.trainer import Trainer
from utils.utils import get_device, read_parameters, separate_train_val
from metrics.metrics import Accuracy_Metric

from PIL import Image

def load_image(image_name, transformer=None):
    image = Image.open(image_name)
    if transformer == None:
        image = np.array(image).transpose(2, 0, 1)/255
        image = image.astype(np.float32)
    else:
        image = transformer(image)
    return image

def evaluate(model, data):
    out = model(data)
    return out

if __name__ == "__main__":
    configs = read_parameters()
    device = get_device
    ResNet50 = TL_ResNet50(configs["network_parameters"], pretrained = True).to(device)
    ResNet50.eval()
