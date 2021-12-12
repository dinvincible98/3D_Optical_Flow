from os import posix_fadvise
from tokenize import group
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lib import pointnet2_utils as pointutils

class IO_stream():
    def __init__(self,path):

        self.f = open(path,'a')

    def cprint(self,text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()        
    

def motion_ME_np(pred,label):
    '''
    Motion Error Function
    '''

    # error = np.sqrt(np.sum((pred-label)**2,2))        

    # gtmotion_len= np.sqrt(np.sum(label*label,2))    # B, N

    acc = 0
    
    # Two ACC standard
    # acc1 = np.sum(np.logical_or((error<=0.01)*mask, (error/gtmotion_len<=0.01)*mask),axis=1)
    # acc2 = np.sum(np.logical_or((error<=0.02)*mask, (error/gtmotion_len<=0.02)*mask),axis=1)

    # mask_sum = np.sum(mask,1)

    # acc1 = acc1[mask_sum>0] / mask_sum[mask_sum>0]
    # acc1 = np.mean(acc1)

    # acc2 = acc2[mask_sum>0] / mask_sum[mask_sum>0]
    # acc2 = np.mean(acc2)

    # ME = np.sum(error*mask,1)[mask_sum>0] / mask_sum[mask_sum>0]
    # ME = np.mean(ME)

    ME = np.mean(np.sqrt(np.sum((pred-label)**2,2)))   
    print(f"ME shape is: {ME.shape}, ME: {ME}")

    if ME < 0.005:
        acc += 1




    return ME, acc

####################################################################
def square_dist(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    
    Parameter
    ---------
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    
    Return
    ------
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).view(B,N,1)
    dist += torch.sum(dst**2, -1).view(B,1,M)

    return dist


# def farthest_point_sample(xyz, n_point):
#     device = xyz.device             # cuda
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B,n_point,dtype=torch.long).to(device)
#     distance = torch.ones(B,N).to(device) * 1e10
#     farthest = torch.randint(0,N,(B,),dtype=torch.long).to(device)
#     batch_indices = torch.arange(B,dtype=torch.long).to(device)

#     for i in range(n_point):
#         centroids[:,i] = farthest
#         centroid = xyz[batch_indices,farthest,:].view(B,1,3)
#         dist = torch.sum((xyz - centroid)**2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance,-1)[1]
    

#     return centroids





def query_ball_points(radius,n_sample,xyz,new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1,1,N).repeat([B,S,1])
    sqr_dist = square_dist(new_xyz,xyz)
    group_idx[sqr_dist > radius**2] = N
    mask = group_idx != N
    cnt = mask.sum(dim=-1)
    group_idx = group_idx.sort(dim=-1)[0][:,:,:n_sample]
    group_first = group_idx[:,:,0].view(B,S,1).repeat([1,1,n_sample])
    mask = group_idx==N
    group_idx[mask] = group_first[mask]


    return group_idx, cnt




class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_pts, radius, n_samples, in_channels, mlp , mlp2=None, group_all=False):
        super().__init__()
        self.n_pts = n_pts
        self.radius = radius
        self.n_samples = n_samples
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()

        last_channel = in_channels + 3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel,1,bias=False)) 
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2 is not None:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel,out_channel,1,bias=False),
                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        
        if group_all:
            self.query_and_group = pointutils.GroupAll()
        else:
            self.query_and_group = pointutils.QueryAndGroup(radius,n_samples)

    def forward(self, xyz, points):
        # print("Start Set Conv!")
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0,2,1).contiguous()

        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t,self.n_pts)    # [B,N]
            new_xyz = pointutils.gather_operation(xyz,fps_idx)          #[B,C,N]
        else:
            new_xyz = xyz
        
        new_points = self.query_and_group(xyz_t,new_xyz.transpose(2,1).contiguous(), points)  #[B,3+C,N,S]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]                # batch normal 2d
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points,-1)[0]  

        if self.mlp2_convs:
            for i, conv in enumerate(self.mlp2_convs):
                new_points = F.relu(conv(new_points))
        

        return new_xyz, new_points



class FlowEmbedding(nn.Module):
    def __init__(self,radius, n_sample, in_channels, mlp, pooling='max', corr_func='concat', knn=True):
        super().__init__()
        self.radius = radius
        self.n_sample = n_sample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        if corr_func is 'concat':
            last_channel = in_channels*2 + 3
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel,out_channel,1,bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        

    def forward(self,pos1,pos2,feature1,feature2):
        # print("Start Flow Embedding!")
        pos1_t = pos1.permute(0,2,1).contiguous()
        pos2_t = pos2.permute(0,2,1).contiguous()

        B, N, C = pos1_t.shape          # batch size, n points, channel

        if self.knn:
            _, idx = pointutils.knn(self.n_sample,pos1_t,pos2_t)
        else:
            # If the ball neighborhood points are less than n sample, then use knn neighborhood points
            idx, cnt = query_ball_points(self.radius,self.n_sample,pos1_t,pos2_t)

            # use knn to find nearest point
            _, idx_knn = pointutils.knn(self.n_sample,pos1_t,pos2_t)
            cnt = cnt.view(B,-1,1).repeat(1,1,self.n_sample)
            idx = idx_knn[cnt > (self.n_sample-1)]

        pos2_grouped = pointutils.grouping_operation(pos2,idx)          # [B,3,N,S]
        # print(pos2_grouped)
        pos_diff = pos2_grouped - pos1.view(B,-1,N,1)                   # [B,3.N,S]


        feature2_grouped = pointutils.grouping_operation(feature2,idx)  # [B,C,N,S]

        if self.corr_func == 'concat':
            feature_diff = torch.cat([feature2_grouped, feature1.view(B,-1,N,1).repeat(1,1,1,self.n_sample)],dim=1)
        
        feature1_new = torch.cat([pos_diff,feature_diff],dim=1)   # [B, 2*C+3,N,S]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feature1_new = F.relu(bn(conv(feature1_new)))

        feature1_new = torch.max(feature1_new,-1)[0]        # [B, mlp[-1], n_point]

        return pos1, feature1_new



class PointNetSetUpConv(nn.Module):
    def __init__(self,n_sample, radius, f1_channel, f2_channel, mlp, mlp2, knn=True):
        super().__init__()
        self.n_sample = n_sample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3

        if mlp is not None:
            for out_channel in mlp:
                self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel,out_channel,1,bias=False),
                                                    nn.BatchNorm2d(out_channel),
                                                    nn.ReLU(inplace=False)))
                last_channel = out_channel
        
        if len(mlp) is not 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel += f1_channel
        
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel,out_channel,1,bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))

            last_channel = out_channel
    

    def forward(self, pos1, pos2, feature1, feature2):
        # print("Start Up Conv!")
        pos1_t = pos1.permute(0,2,1).contiguous()       # [B,C,N]
        pos2_t = pos2.permute(0,2,1).contiguous()

        B, C, N = pos1.shape

        if self.knn:
            _, idx = pointutils.knn(self.n_sample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_points(self.radius,self.n_sample,pos1_t,pos2_t)
        
        pos2_grouped = pointutils.grouping_operation(pos2,idx)
        pos_diff = pos2_grouped - pos1.view(B,-1,N,1)           # [B,3,N1,S]

        feature2_grouped = pointutils.grouping_operation(feature2,idx)
        feature_new = torch.cat([feature2_grouped,pos_diff],dim=1)          # [B,C1+3,N1,S]

        
        for conv in self.mlp1_convs:
            feature_new = conv(feature_new)

        # max pooling
        feature_new = feature_new.max(-1)[0]

        # concatenate feature in early layer
        if feature1 is not None:
            feature_new = torch.cat([feature_new,feature1],dim=1)
        
        for conv in self.mlp2_convs:
            feature_new = conv(feature_new)

        
        return feature_new


class PointNetFeaturePropogation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel,out_channel,1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        
    def forward(self, pos1, pos2, feature1, feature2):
        '''
        Parameter:
        ---------
            pos1: input points position data, [B,C,N]
            pos2: sampled input points position data, [B,C,S]
            feature1: input points data, [B,D,N] 
            feature2: sampled input points data, [B,D,S]
        
        Return:
        ------
            new_points: unsampled point data, [B,D',N]

        '''
        # print("Start Feature Propagation!")
        pos1_t = pos1.permute(0,2,1).contiguous()
        pos2_t = pos2.permute(0,2,1).contiguous()
        # print(pos1_t.shape)
        # print(pos2_t.shape)

        B, C, N = pos1.shape

        dists, idx = pointutils.three_nn(pos1_t,pos2_t)
        dists[dists < 1e-10] = 1e-10

        weight = 1.0 / dists
        weight = weight / torch.sum(weight,-1, keepdim=True)    #[B,N,3]

        interpolated_feature = torch.sum(pointutils.grouping_operation(feature2,idx)*weight.view(B,1,N,3),dim=-1)   #[B,C,N,3]

        if feature1 is not None:
            feature_new = torch.cat([interpolated_feature,feature1],1)
        else:
            feature_new = interpolated_feature
        
        for idx, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[idx]
            feature_new = F.relu(bn(conv(feature_new)))
        

        return feature_new
