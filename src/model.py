from ast import increment_lineno
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.utils import FlowEmbedding, PointNetFeaturePropogation, PointNetSetAbstraction, PointNetSetUpConv




class MotionNet3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        # conv layer
        self.sa1 = PointNetSetAbstraction(n_pts=1024,radius=0.004,n_samples=16,in_channels=3,mlp=[32,32,64],group_all=False)
        self.sa2 = PointNetSetAbstraction(n_pts=256,radius=0.008,n_samples=16,in_channels=64,mlp=[64,64,128],group_all=False)
        self.sa3 = PointNetSetAbstraction(n_pts=64,radius=0.016,n_samples=8,in_channels=128,mlp=[128,128,256],group_all=False)
        self.sa4 = PointNetSetAbstraction(n_pts=16, radius=0.032,n_samples=8,in_channels=256,mlp=[256,256,512],group_all=False)

        # flow embedding layer
        self.fe = FlowEmbedding(radius=0.1,n_sample=64, in_channels=128, mlp=[128,128,128], pooling='max', corr_func='concat')

        # conv up layer
        self.su1 = PointNetSetUpConv(n_sample=8,radius=0.02,f1_channel=256,f2_channel=512,mlp=[],mlp2=[256,256])
        self.su2 = PointNetSetUpConv(n_sample=8,radius=0.01,f1_channel=128+128,f2_channel=256,mlp=[128,128,256],mlp2=[256])
        self.su3 = PointNetSetUpConv(n_sample=8,radius=0.005,f1_channel=64,f2_channel=256,mlp=[128,128,256],mlp2=[256])

        # Feature Propagation
        self.fp = PointNetFeaturePropogation(in_channel=256+3, mlp=[256,256])

        self.conv1 = nn.Conv1d(256,128,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128,3,kernel_size=1,bias=True)
        

    def forward(self,pc1,pc2,feature1,feature2):
        # Learn deep point cloud features
        l1_pc1, l1_feature1 = self.sa1(pc1,feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1,l1_feature1)

        l1_pc2, l1_feature2 = self.sa1(pc2,feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2,l1_feature2)
        
        # Learn geometric relationship
        _, l2_feature_new = self.fe(l2_pc1,l2_pc2,l2_feature1,l2_feature2)

        # Learn deep point cloud features
        l3_pc1, l3_feature1 = self.sa3(l2_pc1,l2_feature_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1,l3_feature1)

        # Up point sample and propagate feature
        l3_fnew1 = self.su1(l3_pc1,l4_pc1,l3_feature1,l4_feature1)
        l2_fnew1 = self.su2(l2_pc1,l3_pc1,torch.cat([l2_feature1,l2_feature_new],dim=1),l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1,l2_pc1,l1_feature1,l2_fnew1)

        l0_fnew1 = self.fp(pc1,l1_pc1,feature1,l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)

        return sf


# if __name__ == '__main__':
#     import os
#     import torch

#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     input = torch.randn((3,3,64))
#     label = torch.randn((3,8))
#     model = MotionNet3D()
#     output = model(input,label)

#     print(output.size())







