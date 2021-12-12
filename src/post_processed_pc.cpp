/// \file post_processed_point_cloud.cpp
/// \brief This is the source file for post_processed_point_cloud.cpp


#include <ros/ros.h>
#include <ros/console.h>
#include <pcl_ros/point_cloud.h>


#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>

#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>

#include <string>

#include <optical_flow/motion.h>

static ros::Publisher pc_pub;
static ros::Publisher pc_pub2;
static ros::Publisher pc_pub3;
// static ros::Publisher pc_pub4;
// static ros::Publisher motion_pub;
// static::ros::Publisher motion_pub;


// static float x;
// static float y;
// static float z;

// static float vx, vy, vz;

static geometry_msgs::Pose pose_;
static geometry_msgs::Twist twist_;
static std::string name;

// static optical_flow::motion motion_;
// static optical_flow::motion motion_;


// inline float euclidean_dist_3d(float x1, float y1, float z1, float x2, float y2, float z2)
// {
//     return std::sqrt(pow(x1-x2,2) + pow(y1-y2,2) + pow(z1-z2,2));
// }


// void poseCallback(const gazebo_msgs::ModelStates::ConstPtr & msg)
// {
//     name = msg->name.at(2);
//     // ROS_INFO_STREAM("name: "<<name);

//     pose_.position.x = msg->pose.at(2).position.x;
//     pose_.position.y = msg->pose.at(2).position.y;
//     pose_.position.z = msg->pose.at(2).position.z;

//     twist_.linear.x = msg->twist.at(2).linear.x;
//     twist_.linear.y = msg->twist.at(2).linear.y;
//     twist_.linear.z = msg->twist.at(2).linear.z;

//     twist_.angular.x = msg->twist.at(2).angular.x;
//     twist_.angular.y = msg->twist.at(2).angular.y;
//     twist_.angular.z = msg->twist.at(2).angular.z;


//     // ROS_INFO_STREAM("pose y: "<<pose_.position.y);

//     pose_.orientation.x = msg->pose.at(2).orientation.x;
//     pose_.orientation.y = msg->pose.at(2).orientation.y;
//     pose_.orientation.z = msg->pose.at(2).orientation.z;
//     pose_.orientation.w = msg->pose.at(2).orientation.w;

// //    flag = true;

// }



void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& src)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
//    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;

    pcl::PCLPointCloud2 cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*src, *cloud);

    pcl::PCLPointCloud2ConstPtr cloud_ptr(cloud);

    // Perform VoxelGrid filtering
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud(cloud_ptr);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.6,1.3);
    pass.filter(cloud_filtered);

    // Convert PCL to ROS data type
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(cloud_filtered,output);

    // publish data
    pc_pub.publish(output);

//    ROS_INFO("Published pc!");

}
void cloud_callback2(const sensor_msgs::PointCloud2ConstPtr& src)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
//    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;

    pcl::PCLPointCloud2 cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*src, *cloud);

    pcl::PCLPointCloud2ConstPtr cloud_ptr(cloud);

    // Perform VoxelGrid filtering
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud(cloud_ptr);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.25,-0.05);
    pass.filter(cloud_filtered);

    // Convert PCL to ROS data type
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(cloud_filtered,output);


    // publish data
    pc_pub2.publish(output);

//    ROS_INFO("Published pc!");

}
void cloud_callback3(const sensor_msgs::PointCloud2ConstPtr& src)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
//    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;

    pcl::PCLPointCloud2 cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*src, *cloud);

    pcl::PCLPointCloud2ConstPtr cloud_ptr(cloud);

    // Perform Passthrough filtering
    pcl::PassThrough<pcl::PCLPointCloud2> pass;
    pass.setInputCloud(cloud_ptr);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.12,0.12);
    pass.filter(cloud_filtered);

    // Convert PCL to ROS data type
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(cloud_filtered,output);

    // publish data
    pc_pub3.publish(output);
   ROS_INFO("Published pc!");

}

// void cloud_callback4(const sensor_msgs::PointCloud2ConstPtr& src)
// {
//     pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
// //    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;

//     pcl::PCLPointCloud2 cloud_filtered;

//     // Convert to PCL data type
//     pcl_conversions::toPCL(*src, *cloud);

//     pcl::PCLPointCloud2ConstPtr cloud_ptr(cloud);

//     // Perform Passthrough filtering
//     pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
//     sor.setInputCloud(cloud_ptr);
//     sor.setLeafSize(0.005f,0.005f,0.005f);
//     sor.filter(cloud_filtered);

//     // Convert PCL to ROS data type
//     sensor_msgs::PointCloud2 output;
//     pcl_conversions::fromPCL(cloud_filtered,output);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromPCLPointCloud2(cloud_filtered, *temp);

//     // float temp_x = twist_.angular.x;
//     // float temp_y = twist_.angular.y;
//     // float temp_z = twist_.angular.z;

//     motion_.header.stamp = ros::Time::now();

//     for (size_t i=0;i<temp->size();++i)
//     {
//         pcl::PointXYZ v = temp->points[i];

//         x = v.x;
//         y = v.y;
//         z = v.z;

// //        ROS_INFO_STREAM("Twist: "<<"["<<twist_.linear.x<<", "<<twist_.linear.y<<", "<<twist_.linear.z<<
// //                        ", "<<twist_.angular.x<<", "<<twist_.angular.y<<", "<<twist_.angular.z<<"]");

//         // motion_.pose.x = x;
//         // motion_.pose.y = y;
//         // motion_.pose.z = z;
//         // if(motion_.pose.z < 1.10)
//         // {
//         //     motion_.twist.linear.x = twist_.linear.x;
//         //     motion_.twist.linear.y = twist_.linear.y;
//         //     motion_.twist.linear.z = twist_.linear.z;
//         //     motion_.twist.angular.x = temp_x;
//         //     motion_.twist.angular.y = temp_y;
//         //     motion_.twist.angular.z = temp_z;
//         // }
//         // else
//         // {
//         //     motion_.twist.linear.x = 0.0;
//         //     motion_.twist.linear.y = 0.0;
//         //     motion_.twist.linear.z = 0.0;
//         //     motion_.twist.angular.x = 0.0;
//         //     motion_.twist.angular.y = 0.0;
//         //     motion_.twist.angular.z = 0.0;
//         // }
//         if (z < 1.10)
//         {
//             vx = twist_.linear.x;
//             vy = twist_.linear.y;
//             vz = twist_.linear.z;
//         }
//         else
//         {
//             vx = 0.0, vy = 0.0 , vz = 0.0;
//         }

//         motion_.data.push_back(vx);
//         motion_.data.push_back(vy);
//         motion_.data.push_back(vz);


//     }
//     motion_pub.publish(motion_);



// //    ROS_INFO_STREAM("Pose: "<<"["<<pose_.position.x<<", "<<pose_.position.y<<", "<<pose_.position.z<<"]");
// //    ROS_INFO_STREAM("Twist: "<<"["<<twist_.linear.x<<", "<<twist_.linear.y<<", "<<twist_.linear.z<<"]");
//     // publish data
//     pc_pub4.publish(output);

//     ROS_INFO("Published pc!");

// }


int main(int argc, char** argv)
{
    // Initialize ros
    ros::init(argc,argv,"post_processed_pc");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);

    // ros::Subscriber pose_sub = nh.subscribe("gazebo/model_states",10,poseCallback);
    // motion_pub = nh.advertise<geometry_msgs::Twist>("motion",10);
    // motion_pub.publish(twist_);

    // Subscribe to realsense point cloud
    ros::Subscriber pc_sub = nh.subscribe("camera/depth/color/points",10,cloud_callback);
    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("post_processed_pc",10);

    ros::Subscriber pc_sub2 = nh.subscribe("post_processed_pc",10,cloud_callback2);
    pc_pub2 = nh.advertise<sensor_msgs::PointCloud2>("post_processed_pc2",10);

    ros::Subscriber pc_sub3 = nh.subscribe("post_processed_pc2",10,cloud_callback3);
    pc_pub3 = nh.advertise<sensor_msgs::PointCloud2>("post_processed_pc3",10);

    // ros::Subscriber pc_sub4 = nh.subscribe("post_processed_pc3",10,cloud_callback4);
    // pc_pub4 = nh.advertise<sensor_msgs::PointCloud2>("post_processed_pc4",10);

    // motion_pub = nh.advertise<optical_flow::motion>("motion",10);
    // motion_pub = nh.advertise<optical_flow::motion>("motion",10);



    ros::spin();

    return 0;

}


