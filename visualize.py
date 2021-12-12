from shutil import move
import open3d as o3d
import numpy as np

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch

# pcd = o3d.io.read_point_cloud("./pred_motion/pred_motion_0.pcd")
# np_pcd = np.asarray(pcd.points)
# print(np_pcd)



# pcd2 = o3d.io.read_point_cloud("./pred_motion/pred_motion_1.pcd")
# np_pcd2 = np.asarray(pcd2.points)
# print(np_pcd2)


# np_pcd1 = np_pcd[:2048,:]
# np_pcd2 = np_pcd[2048:,:]

# print(np_pcd1)
# print(np_pcd2)


# diff = np.mean(np_pcd2 - np_pcd1,axis=0)
# print(diff)

# o3d.visualization.draw_geometries([pcd])


def predict():
    model_path = './models/test_model_11.pt'
    model = torch.load(model_path)
    model.eval()
    print("Eval mode")



def show_pcd():
    path = './npz'


    fname = path + '/test_' + str(1)+'.npz'
    data = np.load(fname)
    pc1 = data['pcd1']
    pc2 = data['pcd2']
    motion = data['motion']

    print(pc1)
    print(pc2.shape)
    print(pc2-pc1)

    pc1_x = pc1[:,0]
    pc1_y = pc1[:,1]
    pc1_z = pc1[:,2]


    # point dist
    pc1_pt1 = pc1[0,:]
    # pc1_pt2 = pc1[9,:]

    cnt = 0
    for i in range(1,2100):
        pc1_pt = pc1[i,:]
        dist = np.sqrt(np.sum((pc1_pt-pc1_pt1)**2,axis=0))

        if dist < 0.032:
            cnt+=1
    print(cnt)

    # print(pc1_pt2.shape)
    # dist = np.sqrt(np.sum((pc1_pt2-pc1_pt1)**2,axis=0))
    # print(dist)


    pc2_x = pc2[:,0]
    pc2_y = pc2[:,1]
    pc2_z = pc2[:,2]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111,projection='3d')

    ax.scatter(pc1_x,pc1_z,pc1_y,c="red",label="pcd1")
    ax.scatter(pc2_x,pc2_z,pc2_y,c="blue",label="pcd2")

    ax.set_title("Pointclouds")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")

    plt.show()


    # movement = np.array(movement)
    # # print(cnt)
    # print(movement.shape)

    # x = movement[:,0]
    # print(x.shape)
    # y = movement[:,1]
    # z = movement[:,2]

    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111,projection='3d')

    # ax.scatter(x,y,z,c="blue")
    # ax.set_title("Groundtruth Movement")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # plt.show()



def vis_motion(motion_list,num_iter):
    path = './npz'
    movement = []

    # 1 - 100 groundtruth movement 
    for i in range(num_iter):
        fname = path + '/test_' + str(i)+'.npz'
        data = np.load(fname)
        motion = data['motion']
        movement.append(motion[0])

    movement = np.array(movement)
    # print(cnt)
    print(movement.shape)

    x = movement[:,0]
    # print(x.shape)
    y = movement[:,1]
    z = movement[:,2]


    motion_arr = np.array(motion_list)
    print(motion_arr.shape)
    x_motion = motion_arr[:,0]
    y_motion = motion_arr[:,1]
    z_motion = motion_arr[:,2]



    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111,projection='3d')

    ax.scatter(x,y,z,c="blue",label="groundtruth")
    ax.scatter(x_motion,y_motion,z_motion,c="red",label="prediction")


    ax.set_title("Groundtruth Movement for 10 Consecutive Frames")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")

    plt.show()
    
    # plt.savefig("./img/compare.png",fig)

# show_pcd()


# vis_groundtruth_motion()
# predict()

