import numpy as np
import os
import argparse

import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import DataLoader

from src.dataset import MotionDataset
from src.model import MotionNet3D
from src.utils import IO_stream
from src.train import train
from src.test import test, predict_single



def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',help='Name of the experiment')
    parser.add_argument('--embedd dims',type=int,default=512,metavar='N',help='dimension of embeddings')
    parser.add_argument('--num_points',type=int,default=2048,help='Point Number[default: 2048]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',help='Size of batch)')
    parser.add_argument('--epochs',type=int,default=50,metavar='N',help='number of episodes to train')
    parser.add_argument('--use_sgd',action='store_true',default=False,help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,help='train the model')
    parser.add_argument('--eval', action='store_true', default=False,help='evaluate the model')
    parser.add_argument('--predict', action='store_true', default=False,help='prediction')
    parser.add_argument('--dataset_path', type=str, default='./npz', metavar='N',help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',help='Pretrained model path')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


    text_io = IO_stream('./log/'+args.exp_name+'.log')
    text_io.cprint(str(args))

    # DataLoader
    train_loader = DataLoader(MotionDataset(n_pts=args.num_points,
                                            root=args.dataset_path,
                                            partition='train'),
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True)

    test_loader = DataLoader(MotionDataset(n_pts=args.num_points,
                                           root=args.dataset_path,
                                           partition='test'),
                                           batch_size=args.test_batch_size,
                                           shuffle=False,
                                           drop_last=False)

    net = MotionNet3D(args).cuda()
    net.apply(weights_init)
    
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print(f"Use {torch.cuda.device_count()} GPUs!")
    
    if args.train:
        train(args,net,train_loader,test_loader,text_io)

    
    if args.eval:
        if args.model_path == "":
            model_path = './models/test_model_'+args.exp_name+'.pt'
        else:
            model_path = args.model_path
        
        net.load_state_dict(torch.load(model_path),strict=False)

        test(args,net,test_loader,text_io)

    if args.predict:
        if args.model_path == "":
            model_path = './models/test_model_'+args.exp_name+'.pt'
        else:
            model_path = args.model_path
        net.load_state_dict(torch.load(model_path),strict=False)

        predict_single(args,net)


if __name__=='__main__':
    main()





