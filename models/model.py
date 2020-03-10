import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

class TL_ResNet50(nn.Module):
    def __init__(self, params_dict, pretrained=False):
        super().__init__()
        self.ResNet50 = models.resnet50(pretrained = pretrained)
        if pretrained == False:
            for parameter in self.ResNet50.parameters():
                parameters.requires_grad = False
        self.input_number = self.ResNet50.fc.in_features
        self.Linear_1 = nn.Linear(self.input_number, params_dict["linear_1"])
        self.ResNet50.fc = self.Linear_1
        self.Linear_2 = nn.Linear(params_dict["linear_1"], params_dict["linear_2"])
        self.Linear_3 = nn.Linear(params_dict["linear_2"], params_dict["linear_3"])
        self.Dropout_1 = nn.Dropout(p = params_dict["dropout_1"])
        self.Dropout_2 = nn.Dropout(p = params_dict["dropout_2"])

    def forward(self, x):
        x = self.ResNet50(x)
        x = torch.relu(x)
        x = self.Dropout_1(x)
        x = self.Linear_2(x)
        x = torch.relu(x)
        x = self.Dropout_2(x)
        x = self.Linear_3(x)
        return x

if __name__ == "__main__":
    pass