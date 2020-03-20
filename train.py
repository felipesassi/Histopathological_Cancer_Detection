import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.model import TL_ResNet50
from datasets.datasets import generate_train_validation_dataloader
from models.controller import Controller
from utils.utils import get_device, read_parameters, separate_train_val
from metrics.metrics import Accuracy_Metric

if __name__ == "__main__":
    df = pd.read_csv("data/data/train/train_labels.csv")
    df_train, df_val = separate_train_val(df)
    configs = read_parameters()
    device = get_device()
    ResNet50 = TL_ResNet50(params_dict = configs["network_parameters"], pretrained = True).to(device)
    train_loader, val_loader = generate_train_validation_dataloader(df_train, 
                                                                    df_val, 
                                                                    configs["train_parameters"]["batch_size"], 
                                                                    "data/data/train/")
    Loss = nn.BCEWithLogitsLoss()
    Optimizer = optim.Adam(ResNet50.parameters(),
                        lr = configs["train_parameters"]["learning_rate"],
                        weight_decay = configs["train_parameters"]["weight_decay"])
    Metrics = Accuracy_Metric()
    Control = Controller(model = ResNet50,
                optimizer = Optimizer,
                loss = Loss,
                metric = Metrics,
                train_data = train_gen,
                validation_data = val_gen,
                epochs = configs["train_parameters"]["epochs"],
                device = device,
                lr_scheduler = None)
    Control.train()