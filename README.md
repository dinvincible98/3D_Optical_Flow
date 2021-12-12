# 3D_Optical_Flow
### This is the final project for MSR at Northwestern University
### Project Goal

The project is focused on using a rgbd camera(Intel Realsense D435i) to capture an object with random motion. By capturing the pointcloud at consecutive timestamps and giving a groundtruth motion, the deep learning model is able to learn the sub-piexl motion(less than 5mm). The whole project is simulation based using ROS and Gazebo.

### Dependencies
#### Software:
* ROS Noetic, Rviz, Gazebo
* PCL, Open3d
* pyrealsense2, NumPy, OpenCV(4.5.1),cv_bridge
* C++, python3(3.8.1)

### Structure of The Project
#### Part 1.Setup simulation environment

#### Part 2.Post-processing raw pointcloud data

#### Part 3.Given random gaussian motion

#### Part 4.Capture pointcloud data and corresponding groundtruth motion

#### Part 5.Training Deep learning model

#### Part 6.Evaluation of the model
