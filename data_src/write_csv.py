import csv
from os import read, write
import os
import collections

import numpy as np

def step1():
    '''
    Extact data from raw data.csv and update time stamp then saved to groundtruth.csv file 
    '''
    with open("./motion.csv", "r") as source:
        reader = csv.reader(source)

        with open("./groundtruth.csv","w") as result:
            writer = csv.writer(result)
            for r in reader:
                row = r[0];
                new_row = row[:4] +'.' + row[4:]
                writer.writerow((new_row,r[45],r[46],r[47],r[48],r[49],r[50]))
    


def step2():
    Dir = './pcd_3'

    dic = {}

    '''
    1: add all pcd files (time stamp) to the map
    '''
    for filename in os.listdir(Dir):
        # new_filename = filename[:8] + filename[14:]     # remove trailing zeros
        # print(new_filename)
        dic[filename] = None
    
    # sort dictonry based on key
    sorted_dic = collections.OrderedDict(sorted(dic.items()))


    '''
    2. read csv file, for every row find its corresponding time stamp in map
    '''
    with open("./groundtruth.csv", "r") as source:
        reader = csv.reader(source)
        
        for row in reader:
            new_row = row[0] + ".pcd"
            new_x = row[1][:10]             # round up to 9th digit
            new_y = row[2][:10]
            new_z = row[3][:10]
            new_wx = row[4][:10]
            new_wy = row[5][:10]
            new_wz = row[6][:10]

            if new_row in sorted_dic.keys():
                sorted_dic[new_row] = [[new_x],[new_y],[new_z],[new_wx],[new_wy],[new_wz]]
        
        '''
        2278 None-corresponding pcd files
        '''
        # cnt = 0
        # for key,val in sorted_dic.items():
        #     if val is None:
        #         cnt += 1
        # print(f"Non-corresponding cnt: {cnt}")


        
        with open("./label.csv","w") as result:
            writer = csv.writer(result)
            for key, val in sorted_dic.items():
                if val is None: continue                # skip non-corresponding files
                writer.writerow((key,val[0],val[1],val[2],val[3],val[4],val[5]))

def step3():
    '''
    Update label.csv
    '''
    with open("./label.csv", "r") as source:
        reader = csv.reader(source)
        with open("./label_update.csv","w") as result:
            writer = csv.writer(result)
            for r in reader:
                x = float(r[1][2:-2])
                y = float(r[2][2:-2])
                z = float(r[3][2:-2])
                wx = float(r[4][2:-2])
                wy = float(r[5][2:-2])
                wz = float(r[6][2:-2])
                writer.writerow((r[0],x,y,z,wx,wy,wz))



def step4():
    '''
    Match every 2 pcd files with a mean motion vector between 2 time stamp
    '''
    dic = {}
    name_list = []

    with open("./label_update.csv", "r") as source:
        reader = csv.reader(source)
        print("FOUND")
        with open("./label_update2.csv","w") as result:
            writer = csv.writer(result)
            for r in reader:
                name = r[0]
                x = float(r[1])
                y = float(r[2])
                z = float(r[3])
                wx = float(r[4])
                wy = float(r[5])
                wz = float(r[6])

            
                dic[name] = [x,y,z,wx,wy,wz]
                name_list.append(name)
            
            sorted_dic = collections.OrderedDict(sorted(dic.items()))
            
            # print(len(sorted_dic))

            for i in range(0,len(name_list)-1):
                file1 = name_list[i]
                file2 = name_list[i+1]

                mean_x = np.round(sorted_dic[file1][0],6)
                mean_y = np.round(sorted_dic[file1][1],6)
                mean_z = np.round(sorted_dic[file1][2],6)
                mean_wx = np.round(sorted_dic[file1][3],6)
                mean_wy = np.round(sorted_dic[file1][4],6)
                mean_wz = np.round(sorted_dic[file1][5],6)

                # print(mean_x)

                writer.writerow((file1,file2,mean_x,mean_y,mean_z,mean_wx,mean_wy,mean_wz))
                
    print("Finished")

import open3d as o3d

def step5():
    '''
    Create .npz file 
    '''
    cnt = 0
    test_cnt = 0
    print("Start!")
    with open("./label_update2.csv", "r") as source:
        reader = csv.reader(source)
        for r in reader:
            name1 = r[0]
            name2 = r[1]
            vx = float(r[2])
            vy = float(r[3])
            vz = float(r[4])
            wx = float(r[5])
            wy = float(r[6])
            wz = float(r[7])

            # pcd_name1 = name1[:8] + "000000.pcd"
            # pcd_name2 = name2[:8] + "000000.pcd"
            
            # print(name1)
            # print(name2)
            pcd1 = o3d.io.read_point_cloud("./pcd_3/" + name1)
            pcd1_pos = np.asarray(pcd1.points)
            # l.append(len(pcd1_pos))

            # print(pcd_name1)
            pcd2 = o3d.io.read_point_cloud("./pcd_3/" + name2)
            pcd2_pos = np.asarray(pcd2.points)
            # l.append(len(pcd2_pos))

            pcd1_size = len(pcd1_pos)
            pcd2_size = len(pcd2_pos)
            # print(f'pcd1: {pcd1_size}')
            # print(f'pcd2: {pcd2_size}')

            # motion vector [2100 x 3]
            motion =  np.array([[vx, vy, vz]])
            # motion = np.repeat(motion,2100,axis=0)
            # print(motion.shape)


            if pcd1_size==pcd2_size and pcd1_size>4000 and pcd2_size>4000:
                # print(cnt)
                # create background with 0 motion vector
                motion_update = np.zeros((pcd1_size,3))
                # update moving pc
                    
                for i in range(pcd1_size):
                    if pcd1_pos[i][2] < 1.0:
                        motion_update[i] = motion

                indices_1 = np.random.choice(pcd1_pos.shape[0],4000,replace=False)
                sampled_pcd1 = pcd1_pos[indices_1]
                motion_update = motion_update[indices_1]       # select random motion 
                # print(motion_update.shape)
                # print("hello2")
                # indices_2 = np.random.choice(pcd2_pos.shape[0],8192,replace=False)
                sampled_pcd2 = pcd2_pos[indices_1]
                # print(sampled_pcd2.shape)

                # Set color to black [255,255,255] / 255 (normalize)
                color1 = np.ones((4000,3),dtype='float')         
                color2 = np.ones((4000,3),dtype='float')

                pc1_depth = sampled_pcd1[:,-1]
                # print(pc1_depth.shape)
                # pc1_foreground_depth = np.zeros_like(pc1_depth)
                # for i in range(len(pc1_foreground_depth)):
                valid_mask1 = np.ones_like(pc1_depth,dtype=bool)

                # Depth mask: check if there is occlusion
                valid_mask2 = (sampled_pcd1[:,-1] - sampled_pcd2[:,-1]) > -1e-2
                
                mask =  valid_mask1 & valid_mask2
                # print(mask)


                # print(sampled_pcd2.shape)
                # print(len(sampled_pcd2))
                # print("hello")
                if cnt < 28000:
                    train_path = os.path.join('./data/npz_3/'+'train_'+str(cnt)+'.npz')
                    np.savez_compressed(train_path, pcd1 = sampled_pcd1, 
                                                    pcd2 = sampled_pcd2, 
                                                    color1 = color1, 
                                                    color2 = color2, 
                                                    motion = motion_update, 
                                                    mask = mask)
                    cnt += 1
                    # print('Saved')
                else:
                    test_path = os.path.join('./data/npz_3/'+'test_'+str(test_cnt)+'.npz')
                    np.savez_compressed(test_path, pcd1 = sampled_pcd1, 
                                                   pcd2 = sampled_pcd2, 
                                                   color1 = color1, 
                                                   color2 = color2,  
                                                   motion = motion_update, 
                                                   mask = mask)
                    test_cnt += 1


    print(f'Training set: {cnt}')
    print(f'Testing set {test_cnt}')


    print("Finished")
            # o3d.visualization.draw_geometries([pcd_file1]);

            





def main():
    step5()



if __name__ == '__main__':
    try:
        main()
    except:
        KeyboardInterrupt