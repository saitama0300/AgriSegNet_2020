import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, Sigmoid, Softmax
from torch.nn import functional as F
from torch.autograd import Variable


class Attention(nn.Module):

    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
        
    def forward(self,x1):

    
        o2 = self.layer1(x1)
        op = self.layer2(o2)
        op = self.layer3(op)
        return op
