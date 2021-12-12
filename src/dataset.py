from typing_extensions import ParamSpec
import numpy as np
from torch.utils.data import Dataset

import os
import glob
import h5py


class MotionDataset(Dataset):
    def __init__(self, n_pts=2048, root='./npz',partition='train'):
        self.n_pts = n_pts
        self.root = root
        self.partition = partition

        if self.partition == 'train':
            self.data_path = glob.glob(os.path.join(self.root,'train*.npz'))
        else:
            self.data_path = glob.glob(os.path.join(self.root,'test*.npz'))
    
        self.cache = {}                         # store all index
        self.cache_size = 50000

        self.data_path = [d_path for d_path in self.data_path]

        print(len(self.data_path))

    def __getitem__(self, index):
        if index in self.cache:
            pcd1 , pcd2 ,color1, color2, motion, mask = self.cache[index]
        else:
            fn = self.data_path[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pcd1 = data['pcd1'].astype('float32')
                pcd2 = data['pcd2'].astype('float32')
                color1 = data['color1'].astype('float32')
                color2 = data['color2'].astype('float32')
                motion = data['motion'].astype('float32')
                mask = data['mask']
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pcd1, pcd2, color1, color2, motion, mask)


        if self.partition == 'train':
            # Choose random points
            sample_idx1 = np.random.choice(pcd1.shape[0],self.n_pts,replace=False)
            sample_idx2 = np.random.choice(pcd2.shape[0],self.n_pts,replace=False) 
            # print(self.n_pts)   

            pcd1 = pcd1[sample_idx1,:]
            pcd2 = pcd2[sample_idx2,:]
            color1 = color1[sample_idx1,:]
            color2 = color2[sample_idx2,:]
            motion = motion[sample_idx1,:]
            mask = mask[sample_idx1]
        else:
            # Test does not require to choose random pts
            pcd1 = pcd1[:self.n_pts,:]
            pcd2 = pcd2[:self.n_pts,:]
            color1 = color1[:self.n_pts,:]
            color2 = color2[:self.n_pts,:]
            motion = motion[:self.n_pts,:]
            mask = mask[:self.n_pts]
        
        # Get mean xyz
        pcd1_center = np.mean(pcd1,axis=0)
        # Pos relative to center  
        pcd1 -= pcd1_center
        pcd2 -= pcd1_center

        return pcd1,pcd2,color1,color2,motion,mask

    
    def __len__(self):
        return len(self.data_path)
    




def main():
    da = MotionDataset(partition='train')
    pcd1, pcd2, color1, color2, motion, mask = da.__getitem__(1)
    
    pcd3 = np.concatenate((pcd1,pcd2),axis=0)
    print(pcd3.shape)





if __name__ == '__main__':
    try:
        main()
    except:
        raise KeyboardInterrupt