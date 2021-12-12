Run:

1. $ roslaunch optical_flow view_d435_model_rviz_gazebo.launch 
2. $ rosrun optical_flow post_processed_node
3. $ rosrun optical_flow simulation_node




Week 2 (Collect data from simulation)
1. save depth image and calulate pixel in meters.   (u - cx) * depth / fx       (Done)
2. Modify the simulation (Add random noise to x, y, z, row, pitch, yaw with 
maximum displacement of 10 pixels distance)				  						(Done)
3. Save rosbags (50,000 pcd files and ground truth motion) 						(Done saved 68000+ pcd files)
5. Post-processed pcd files (Extract xyz info)									(Done)
6. Create label (one motion vector for two input pcd files)    					(Ongoing)
4

Week 3 (Label Data)
1. Update time stamp in groundtruth.csv file (Iterate through all time stamp and add a '.' at index 4).
2. Iterate through all pcd_xyz files, use map to store all .pcd files name. Iterate through all row (time stamp) in groundtruth.csv. If map finds time stamp, save the time stamp and corresponding linear and rotational velocity vector (6 in total) into a label.csv. 													(Done, found 2278 non-corresponding pcd files)
3. Iterate through all pcd_xyz files, save every pair of two pcd filenames into same row in train.csv file. Create a map using label.csv. (time stamp corresponding to motion vector). Iterate through every row of train.csv, if map find filename1(row[0]) and filename2(row[1]), store an average value of two corresponding motion vector(np.mean(map[row[0]] + map[row[1]])). Saved all data (3 colums) into train.csv.												(Done)
4. Split 20% for test dataset and 10% for validaton dataset.


Week 4
1. Turn off the noise (Check pointcloud noise so urce code)				(Done, librealsense gazebo uses depth image to fill pointcloud)
2. Redo simulation														(Done)
	a. Get x,y,z locations of every points
	b. Calculate 3d euclidean distance between each point and center point and cube.
	c. The linear velocity is the same for all points, using 3d euclidean distance and motion vector of center point to calculate motion vector of each point.
	d. Create customized ros msg. Format:
	geometry_msgs/Point:
		x
		y
		z
	geometry_msgs/Twist:
		vx
		vy
		vz
		wx
		wy
		wz
	e. Record rosbag then convert to .csv file.     
	

