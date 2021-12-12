import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from typing import Tuple
import pointnet2_cuda as pointnet2



class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, n_point:int) -> torch.Tensor:
        '''
        Parameters
        ----------
        xyz: torch.Tensor
            (B, N, 3) tensor where N > n_point 
        n_point: int32
            number of features in the sampled set
        
        Returns
        ------- 
            (B,n_point) tensor containing the set
        
        '''
        assert xyz.is_contiguous()

        B,N,_ = xyz.size()
        output = torch.cuda.IntTensor(B, n_point)
        temp = torch.cuda.FloatTensor(B,N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B,N,n_point,xyz,temp,output)

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None,None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        ctx:
        features: torch.Tensor
                (B, C, N) tensor
        idx: torch.Tensor
                (B, n_point) tensor of the features to gather

        Returns
        -------
        output: torch.Tensor
                (B, C, n_point) tensor       
        '''
        
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, n_point = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B,C,n_point)

        pointnet2.gather_points_wrapper(B,C,N,n_point,features,idx,output)

        ctx.for_backwards = (idx,C,N)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, n_point = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B,C,N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B,C,N,n_point,grad_out_data,idx,grad_features.data)

        return grad_features, None

gather_operation = GatherOperation.apply



class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
        assert features.is_contiguous()
        assert idx.is_contiguous()
        idx = idx.int()
        # print(idx.size())
        B, n_features, n_sample = idx.size()
        _, C, N = features.size()
        
        output = torch.cuda.FloatTensor(B,C,n_features,n_sample)
        pointnet2.group_points_wrapper(B,C,N,n_features,n_sample,features,idx,output)
        
        ctx.for_backwards = (idx,N)

        return output
    
    @staticmethod
    def backward(ctx, grad_out:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx, N = ctx.for_backwards

        B, C, n_points, n_sample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B,C,N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B,C,N,n_points,n_sample,grad_out_data,idx,grad_features.data)

        return grad_features, None

grouping_operation = GroupingOperation.apply

class KNN(Function):
    @staticmethod
    def forward(ctx, k:int, unknown:torch.Tensor, known:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)

        dist2 = torch.cuda.FloatTensor(B,N,k)
        idx = torch.cuda.IntTensor(B,N,k)

        pointnet2.knn_wrapper(B,N,m,k,unknown,known,dist2,idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None

knn = KNN.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown:torch.Tensor, known:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N , _ = unknown.size()
        m = known.size(1)
        # print(B)
        # print(N)
        # print(m)


        dist2 = torch.cuda.FloatTensor(B,N,3)
        idx = torch.cuda.IntTensor(B,N,3)

        pointnet2.three_nn_wrapper(B,N,m,unknown,known,dist2,idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

        
three_nn = ThreeNN.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius:float, n_sample:int, xyz:torch.Tensor, new_xyz:torch.Tensor) -> torch.Tensor:
        
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        n_point = new_xyz.size(1)

        idx = torch.cuda.IntTensor(B, n_point, n_sample).zero_()

        pointnet2.ball_query_wrapper(B,N,n_point,radius,n_sample,new_xyz,xyz,idx)
        
        
        return idx
    @staticmethod
    def backward(ctx, a=None):
        return None,None,None,None
     
ball_query = BallQuery.apply



class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz:torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        '''
        :param xyz: (B,N,3) xyz coordinates of the features
        :param new_xyz: N/A
        :param features: (B,C,N) descriptors of the features
        :return:
            new_features: (B,C+3,1,N)  
        '''
        
        grouped_xyz = xyz.transpose(1,2).unsqueeze(2)
        
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        
        return new_features


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, n_sample: int, use_xyz:bool=True):
        super().__init__()
        self.radius = radius
        self.n_sample = n_sample
        self.use_xyz = use_xyz

    
    def forward(self, xyz:torch.Tensor, new_xyz:torch.Tensor, features:torch.Tensor=None) -> Tuple[torch.Tensor]:
        idx = ball_query(self.radius,self.n_sample,xyz,new_xyz)
        xyz_trans = xyz.transpose(1,2).contiguous()
        
        # print(xyz.shape)
        # print(xyz_trans.shape)
        
        grouped_xyz = grouping_operation(xyz_trans, idx)
        # print(grouped_xyz.shape)

        grouped_xyz -= new_xyz.transpose(1,2).unsqueeze(-1)     # (B, 3, n_point, n_sample)

        if features is not None:
            grouped_features = grouping_operation(features,idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz,grouped_features],dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have no features and  use xyz as a feature!"
            new_features = grouped_xyz
        
        return new_features






