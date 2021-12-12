import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# import open3d as o3d
from src.utils import motion_ME_np, IO_stream
from visualize import vis_motion


def predict_single(args,net:nn.Module):
    net.eval()
    path = '/home/mingqing/3d_optical_flow/npz'
    for i in range(1000):
        fname = path + '/test_' + str(i)+'.npz'
        data = np.load(fname)
        pc1 = data['pcd1'].astype('float32')
        pc2 = data['pcd2'].astype('float32') 
        color1 = data['color1'].astype('float32')
        color2 = data['color2'].astype('float32')
        motion = data['motion'].astype('float32')
        mask = data['mask']
        
        # Get mean xyz
        pcd1_center = np.mean(pc1,axis=0)
        # Pos relative to center  
        pc1 -= pcd1_center
        pc2 -= pcd1_center


        pc1 = torch.from_numpy(pc1)
        pc2 = torch.from_numpy(pc2)
        color1 = torch.from_numpy(color1)
        color2 = torch.from_numpy(color2)
        motion = torch.from_numpy(motion)
        mask = torch.from_numpy(mask)

        # print(f"pc1 shape: {pc1.shape}")
        # print(f"pc2 shape: {pc2.shape}")
        # print(f"color1 shape: {color1.shape}")
        # print(f"color2 shape: {color2.shape}")
        # print(f"motion shape: {motion.shape}")
        # print(f"mask shape: {mask.shape}")

        # cuda type
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        motion = motion.cuda()
        mask = mask.cuda().float()

        motion_pred = net(pc1,pc2,color1,color2).permute(0,2,1)

        print(f"At Index: {i}, motion_pred: {motion_pred}")






def test_one_epoch(args, net:nn.Module, test_loader):
    net.eval()

    total_loss = 0
    total_me = 0
    total_acc = 0
    # total_acc3d_2 = 0
    num_examples = 0

    motion_list = []

    # test first 10 samples
    for idx, data in tqdm(enumerate(test_loader),total=len(test_loader)):
        if idx < 3:
            pc1, pc2, color1, color2, motion, mask = data

            # print(f"pc1 shape: {pc1.shape}")
            # print(f"pc2 shape: {pc2.shape}")
            # print(f"color1 shape: {color1.shape}")
            # print(f"color2 shape: {color2.shape}")
            # print(f"motion shape: {motion.shape}")
            # print(f"mask shape: {mask.shape}")

            # print(pc1.shape)
            pc1 = pc1.cuda().transpose(2,1).contiguous()
            pc2 = pc2.cuda().transpose(2,1).contiguous()
            color1 = color1.cuda().transpose(2,1).contiguous()
            color2 = color2.cuda().transpose(2,1).contiguous()
            motion = motion.cuda()
            mask = mask.cuda().float()

            batch_size = pc1.size(0)
            num_examples += batch_size

            motion_pred = net(pc1,pc2,color1,color2).permute(0,2,1)

            # print(f'motion is: {motion.shape}')
            # print(f'motion_pred is: {motion_pred.shape}')
            # loss = torch.mean(mask*torch.sum((motion_pred - motion)**2,-1) / 2.0)
            loss = torch.mean(torch.sum((motion_pred - motion)**2,1))

            print(f"At index:  {idx}")
            print(f"Shape motion: {motion.shape}, Result: {motion[0]}")
            print(f"Shape motion pred: {motion_pred.shape}, Result: {motion_pred[0]}")



            motion_pred_np = motion_pred.detach().cpu().numpy()

            motion_pred_np = motion_pred_np.reshape(2048,3)

            motion_pred_np = np.sum(motion_pred_np,axis=0)/2048

            print(f"motion_pred_np shape: {motion_pred_np.shape}, motion pred: {motion_pred_np}")
            
            motion_list.append(motion_pred_np)




            me_3d, acc = motion_ME_np(motion_pred.detach().cpu().numpy(),
                                    motion.detach().cpu().numpy())
                                                    # mask.detach().cpu().numpy())

        
            
            # Sum of errors
            total_me += me_3d * batch_size
            total_acc += acc * batch_size
            # total_acc3d_2 += acc_3d_2 * batch_size
            total_loss += loss.item() * batch_size
        

            mean_loss = total_loss*1.0/num_examples
            mean_me = total_me*1.0/num_examples
            mean_acc = total_acc*1.0/num_examples
            # mean_acc3d2 = total_acc3d_2*1.0/num_examples
        else:   break


    return mean_loss,mean_me,mean_acc,motion_list



def test(args,net:nn.Module,test_loader,text_io:IO_stream):
    test_loss, me, acc, motion_list = test_one_epoch(args,net,test_loader)

    # first 10 pcd
    vis_motion(motion_list,num_iter=3)


    text_io.cprint('==FINAL TEST==')
    text_io.cprint('mean test loss: %f\t ME 3d: %f\t ACC 3d: %f' %(test_loss,me,acc))


