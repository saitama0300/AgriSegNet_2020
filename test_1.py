import sys
sys.path.append("../")

import os
import numpy as np
from DataLoader2 import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torchvision.utils import save_image
import argparse
from AttnModel import *
from utils1 import *
import matplotlib.pyplot as plt
from PIL import Image

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    root_dir = 'G:\Tanmay\Agrivision\Agriculture-Vision'

   
    val_set = ImageDataset('val',root_dir,1)

    val_loader = DataLoader(val_set,
                            batch_size=1,
                            num_workers=0,
                            shuffle=True)

    def to_var(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return x

    def evaluation_1(net):

        total_score,mIoU = 0,0
        accuracy, f1=0,0
        count=0

        net.eval()
        totalImages2=len(val_loader)
        conf_mat = torch.zeros((7,7))
        tot = torch.zeros((7))
        
        for i, (image1, image2, image3, v, _, filename) in enumerate(val_loader):
            
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()
            preds = net(image1,image2,image3,1)
    
            preds = nn.Softmax()(preds)
            pred = torch.argmax(preds,1)
                #print(torch.max(preds),torch.min(preds),torch.max(target))
                #print(preds)
            target = one_hot(v.long(), num_classes=7,device=v.device, dtype=v.dtype)
            inp = one_hot(pred.long(), num_classes=7,device=pred.device, dtype=pred.dtype)

            accuracy_pixel(target.cpu(),inp.cpu(),conf_mat,tot)
            
            total_score += dice_coeff(v.cpu(),pred.cpu())
            count+=1

            '''
            for k in range(image1.size(0)):
                t = preds[k,:,:].data.cpu().numpy()
                im = Image.fromarray(np.uint8(t))
                    
                im.save(filename[k],"PNG")
            '''
            
        
            printProgressBar(i+1, totalImages2,
                            prefix="[Test Check] : ",
                            length=45,
                            suffix="Metric: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(conf_mat[0][0]/tot[0],conf_mat[1][1]/tot[1],conf_mat[2][2]/tot[2],conf_mat[3][3]/tot[3],conf_mat[4][4]/tot[4],conf_mat[5][5]/tot[5],conf_mat[6][6]/tot[6]))
            
        #Confusion Matrix
        print(conf_mat[0]/tot[0])
        print(conf_mat[1]/tot[1])
        print(conf_mat[2]/tot[2])
        print(conf_mat[3]/tot[3])
        print(conf_mat[4]/tot[4])
        print(conf_mat[5]/tot[5])
        print(conf_mat[6]/tot[6])
        print(tot)
        #Mean Dice Score or F1 Score across all the images
        print(total_score/count)
        return 

    def evaluation_2(net):

        def eval_batch(target, preds):

            score,_ = intersection_and_union(preds, target, 7)
            return score,_

        total_score,mIoU = 0,0
        score_list = []
        net.eval()
        count=0
        totalImages2=len(val_loader)
        
        
        for i, (image1, image2, image3, v, _, filename) in enumerate(val_loader):
            
            image1 = image1.cuda()
            image2 = image2.cuda()
            image3 = image3.cuda()
            preds = net(image1,image2,image3,1)
            
            preds = nn.Softmax()(preds)
            pred = torch.argmax(preds,1)
                #print(torch.max(preds),torch.min(preds),torch.max(target))
                #print(preds)
            for k in range(image1.size(0)):
                IoU, iou_list = eval_batch(v[k,:,:].unsqueeze(0),pred[k,:,:].unsqueeze(0))
                total_score += IoU
                score_list.append(iou_list)
                count+=1
            
            printProgressBar(i+1, totalImages2,
                            prefix="[Test Check] : ",
                            length=45,
                            suffix="_")
           
        mIoU = total_score/count
        _  = np.nanmean(score_list,axis=0)
        return total_score/count, _

    net = Model()

    PATH = "G:\Tanmay\Agrivision\AttentionModel\savefiles\checkptDeepLab60.pth"

    mdl = torch.load(PATH)
    net.load_state_dict(mdl['model_state_dict'])

    net.cuda()
    evaluation_1(net)
    print(evaluation_2(net))

