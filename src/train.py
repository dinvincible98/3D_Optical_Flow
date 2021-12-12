import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR

import numpy as np
from tqdm import tqdm

from src.test import test_one_epoch
from src.utils import IO_stream

import time 


def train_one_epoch(args, net, train_loader,opt):
    net.train()
    num_examples = 0
    total_loss = 0

    # Loop train_loader
    for idx, data in tqdm(enumerate(train_loader),total=len(train_loader)):
        pc1, pc2, color1, color2, motion, mask = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        motion = motion.cuda().transpose(2,1).contiguous()
        mask = mask.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size

        # print(pc1.shape)
        # print(pc2.shape)
        # print(color1.shape)
        # print(color2.shape)
        # Predicted motion
        motion_pred = net(pc1,pc2,color1,color2)
        # loss = torch.mean(mask * torch.sum((motion_pred - motion)**2,1) / 2.0)
        loss = torch.mean(torch.sum((motion_pred - motion)**2,1))           # MSE

        # loss = torch.nn.MSELoss(motion_pred,motion)         # L2 mse loss
        loss.backward()


        opt.step()
        total_loss += loss.item() * batch_size
    
    return total_loss*1.0 / num_examples



def train(args,net:nn.Module,train_loader,test_loader,text_io:IO_stream):
    if args.use_sgd:
        print("Use SGD optimizer")
        opt = optim.SGD(net.parameters(),lr=args.lr*100, momentum=args.momentum,weight_decay=1e-4)
    else:
        print("Use Adam optimizer")
        opt = optim.Adam(net.parameters(),lr=args.lr,weight_decay=1e-4)
    
    scheduler = StepLR(opt,10,gamma=0.7)

    best_test_loss = np.inf

    start_time = time.time()
    for epoch in range(args.epochs):
        text_io.cprint('==Epoch: %d, learning rate: %f==' %(epoch,opt.param_groups[0]['lr']))
        train_loss = train_one_epoch(args,net,train_loader,opt)
        text_io.cprint('mean train ME loss: %f' %train_loss)

        test_loss, me, acc  = test_one_epoch(args,net,test_loader)
        text_io.cprint('mean test loss: %f\t ME 3d: %f\t ACC 3d: %f' %(test_loss,me,acc))

        # Keep updating best loss
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            text_io.cprint('Best test loss till now: %f' %test_loss)
            
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), './models/test_model'+args.exp_name+'.pt')
            else:
                torch.save(net.state_dict(), './models/test_model_'+args.exp_name+'.pt')
            text_io.cprint('Saved model!')

        # Update steps(lr) based on epoches
        scheduler.step()
    
    end_time = time.time()
    duration = (end_time-start_time) / 3600         #in hr
    text_io.cprint('Training time spent: %f h'%duration)
    
