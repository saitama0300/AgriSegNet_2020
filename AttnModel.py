from AttentionWeighted import *
from resnext50 import *

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class seg(nn.Module):
    def __init__(self):

        super(seg,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32 , kernel_size=3, padding=1), nn.BatchNorm2d(32),nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 7, kernel_size=1))

    def forward(self,x1):
    
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x3 = self.layer3(x3)
        return x3

class ModelSingle(nn.Module):
    def __init__(self):
        
        super(ModelSingle, self).__init__()
        
        self.segment = seg()

    def forward(self,x,trunk):

        x = trunk(x)
        x = self.segment(x)
        return x

class ModelwithAttn(nn.Module):
    def __init__(self):
        
        super(ModelwithAttn, self).__init__()
        
        #UpSampling as  - F.upsample(self.down1(X), size=X.size()[2:], mode='bilinear')
        self.segm1 = seg()
    
        self.Attn = Attention(128)

    def forward(self,x1,x2,trunk):

        layer1 = trunk(x1)

        #print(layer1.shape,layer2.shape)
       
        wt = self.Attn(layer1)
        wt1 = F.upsample(1-wt, size=x2.size()[2:], mode='bilinear')

        #print(w1.shape, w2.shape)
        f1 = self.segm1(layer1)
        a = F.upsample(wt*f1, size=x2.size()[2:], mode='bilinear')
        #print(f1.shape,f2.shape)
        output = a + (wt1)*x2

        return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.m1= ModelwithAttn()
        self.m2 = ModelSingle()
        self.trunk = model_map['deeplabv3plus_resnet50'](num_classes=128, output_stride=8)
        
    def forward(self,x1,x2,x3,mode):
        
        if mode==1:
            p1 = self.m2(x3,self.trunk)
            p2 = self.m1(x2,p1,self.trunk)
            p3 = self.m1(x1,p2,self.trunk)
            return p3
        else:
            p1 = self.m2(x3,self.trunk)
            p3 = self.m1(x2,p1,self.trunk)
            return p3
        
#n = torch.Tensor(np.ones((1,4,128,128)))
'''
model = Model()
n2 = torch.Tensor(np.ones((2,4,256,256)))
n3 = torch.Tensor(np.ones((2,4,512,512)))
p = model(n2,n2)
print(p.shape)
'''