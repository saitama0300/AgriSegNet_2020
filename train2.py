import os
import numpy as np
from DataLoader2 import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, SGD

import argparse
from AttnModel import *
from utils1 import *
from torch.autograd import Variable
from Losses import *

root_dir = 'G:\Tanmay\Agrivision\Agriculture-Vision'


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.002)
        m.bias.data.fill_(0)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x




def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = args.batch_size
    lr = args.lr

    epoch = args.epochs
    root_dir = args.root
    
    print(' Dataset: {} '.format(root_dir))

    train_set = ImageDataset('train', root_dir,1)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)


    print("~~~~~~~~~~~ Creating the Model ~~~~~~~~~~")

    model = Model()
    model.apply(weights_init)
    model.train()
    
    softMax = nn.Softmax()
    acw_loss = ACW_loss()
    dice_loss = DiceLoss()

    if torch.cuda.is_available():
        model.cuda()
        softMax.cuda()
        acw_loss.cuda()
        dice_loss.cuda()
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60, 1.18e-6)

    #optimizer.load_state_dict(mdl['optimizer_state_dict'])
    #lr_scheduler.load_state_dict(mdl['lr_state_dict'])

    Losses = 0

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        model.train()

        totalImages = len(train_loader)
        count=0
        Losses = 0
        TIoUs = 0
        for j, data in enumerate(train_loader):
            
            image1, image2, image3, labels, v, _ = data
            
            # prevent batchnorm error for batch of size 1
            if image3.size(0) != batch_size:

                continue


            #img = to_var(image1.float())
            img2 = to_var(image2.float())
            img3 = to_var(image3.float())
            segmentation = to_var(labels)

                ################### Train ###################
            model.zero_grad()
            optimizer.zero_grad()
           
            prediction = model(None,img2,img3,0)

            prediction_y = softMax(prediction)

            o1 = segmentation
            o1 = o1.type(torch.LongTensor)
            o1 = to_var(o1)
            p1 = prediction_y
            pred = torch.argmax(p1,1).cuda()
        
            netloss = dice_loss(prediction_y,o1) + acw_loss(prediction, o1)
       
            acc = (pred == o1).float().mean()
            netloss.backward()

            #optimizer.step()
            #lr_scheduler.step(epoch=(start+i+1))

            IoU =0
            Losses+=netloss.cpu().data.numpy()
            TIoUs+=IoU

            count+=1
            
            printProgressBar(j+1, totalImages,
                                prefix="[Training, Batch Size: {}] Epoch: {} ".format(batch_size,i+1),
                                length=20,
                                suffix="Loss : {:.4f}  Acc: {:.4f}".format(Losses/(j+1),acc))

        #ValIoU = validation(net)
        #IoUs.append(ValIoU)
        if(i>=0):
                
                print("Saveing checkpoint..... Mean Loss ={:.4f}".format(Losses/count))
                torch.save({
                            'epoch': i+1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': Losses/count,
                }, "G:\Tanmay\Agrivision\AttentionModel\savefiles\checkptDeepLab{}.pth".format(i+1))

        #print("Validation set IoU at Epoch {} : {}".format(i+1,ValIoU))

    print("---------------------Training Completed-------------------------")
        
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--modelName',default = 'Multi-Scale Attention',type=str)
    parser.add_argument('--root', default = 'G:\Tanmay\Agrivision\Agriculture-Vision', type = str)
    parser.add_argument('--num_workers', default = 0, type = int)
    parser.add_argument('--batch_size',default = 4,type = int)
    parser.add_argument('--epochs',default = 60,type = int)
    parser.add_argument('--lr',default = 0.01,type = float)
    args=parser.parse_args()
    train(args)
