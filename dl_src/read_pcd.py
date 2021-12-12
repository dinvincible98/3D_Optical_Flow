import open3d as o3d
import numpy as np

pcd1 = o3d.io.read_point_cloud('./pcd_new/2594.638000000.pcd')
pcd2 = o3d.io.read_point_cloud('./pcd_new/2594.758000000.pcd')

print(pcd1)
print(pcd2)


pc1 = np.asarray(pcd1.points)
pc2 = np.asarray(pcd2.points)
print(pc1)

a = np.array([[0.5,0.5,0.5]])
b = np.array([[1,1,1]])

print(b.shape)

res = np.empty((1,3))
print(res.shape)

init = np.zeros((len(pc1),3))
print(init.shape)

cnt = 0
for i in range(len(pc1)):
    # if pc[2]<1.0:
    #     res = np.concatenate((res,b),axis=0)
    # else:
    #     res = np.concatenate((res,a),axis=0)
    if pc1[i][2] < 1.0:
        init[i] = a
    else:
        init[i] = b

        
print(cnt)
print(init.shape)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(init)
# o3d.io.write_point_cloud('./test.pcd',pcd)

# pcd_load = o3d.io.read_point_cloud("./test.pcd")
# o3d.visualization.draw_geometries([pcd_load])

# diff = pc2 - pc1
# print(diff)

# dist = np.sqrt(np.sum(diff**2,axis=1))
# print(dist)


# o3d.visualization.draw_geometries([pcd1])