import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

class Accuracy_Metric():
    def __init__(self):
        self.accuracy = 0
        self.accuracy_temp = 0

    def compute_metric(self, y_pred, y_true, batch_size, i):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred <= 0] = 0
        correct = (y_pred == y_true).sum()
        accuracy = 100*correct/batch_size
        self.accuracy_temp = self.accuracy_temp + accuracy
        self.accuracy = self.accuracy_temp/(i + 1)

    def reset_accuracy_value(self):
        self.accuracy = 0
        self.accuracy_temp = 0

    def get_accuracy_value(self):
        return self.accuracy

if __name__ == "__main__":
    pass