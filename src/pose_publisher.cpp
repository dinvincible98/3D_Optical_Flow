#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/PoseStamped.h>
#include <apriltag_ros/AprilTagDetection.h>
#include <apriltag_ros/AprilTagDetectionArray.h>
#include <string>


class Tag
{
public:
    explicit Tag(ros::NodeHandle& nh);


private:
    ros::NodeHandle nh_;

    ros::Subscriber pose_sub;

    ros::Publisher pose_pub;

    void tagCallback(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg);

    double pose_x0,pose_y0,pose_z0;
    double pose_x1,pose_y1,pose_z1;

    std::vector<double> origin;


    int tag0, tag1;

    int n;

    std::string frame_;

    bool tag_flag;

};

Tag::Tag(ros::NodeHandle& nh):nh_(nh)
{
    pose_sub = nh_.subscribe("tag_detections",10,&Tag::tagCallback,this);

    pose_pub = nh_.advertise<geometry_msgs::PoseStamped>("nozzle_pose",10);

    tag_flag = false;

    ros::Rate rate(10);

    ros::Time cur_time,last_time;

    while (ros::ok())
    {

        cur_time = ros::Time::now(); 
        
        if(n==2)
        {
            geometry_msgs::PoseStamped nozzle_pose;
            nozzle_pose.header.frame_id = frame_;
            nozzle_pose.header.stamp = ros::Time::now();
            nozzle_pose.pose.position.x = pose_x0 - origin[0];
            nozzle_pose.pose.position.y = pose_y0 - origin[1];
            nozzle_pose.pose.position.z = pose_z1 - origin[2];

            pose_pub.publish(nozzle_pose);
        }

        last_time = cur_time;

        rate.sleep();

        ros::spinOnce();
    }
    
}



void Tag::tagCallback(const apriltag_ros::AprilTagDetectionArray::ConstPtr &msg)
{
    frame_ = msg->header.frame_id;
    n = msg->detections.size();
    
    origin.reserve(3);

    ROS_INFO_STREAM("size"<<n);

    if (n==0)
    {
        pose_x0 = 0.0;
        pose_y0 = 0.0;
        pose_z1 = 0.0;
    }

    else if(n==1)
    {
        if (msg->detections.at(0).id[0]==0)
        {
            // corresponding to 3d printer's frame
            pose_x0 = msg->detections.at(0).pose.pose.pose.position.x;
            pose_y0 = 0.0;
            pose_z1 = -msg->detections.at(0).pose.pose.pose.position.y;
        }
        else
        {
        
            pose_x0 = 0.0;
            pose_y0 = msg->detections.at(0).pose.pose.pose.position.z;
            pose_z1 = 0.0;
        }
    }
    else if(n==2)
    {
        if(msg->detections.at(0).id[0]==0)
        {
            pose_x0 = msg->detections.at(0).pose.pose.pose.position.x;
            pose_y0 = msg->detections.at(1).pose.pose.pose.position.z;
            pose_z1 = -msg->detections.at(0).pose.pose.pose.position.y;

            origin.push_back(pose_x0);
            origin.push_back(pose_y0);
            origin.push_back(pose_z1);
            // ROS_INFO_STREAM("origin x: "<<origin[0]);
            // ROS_INFO_STREAM("origin y: "<<origin[1]);            
            // ROS_INFO_STREAM("origin z: "<<origin[2]);
        }
        else
        {
            pose_x0 = msg->detections.at(1).pose.pose.pose.position.x;
            pose_y0 = msg->detections.at(0).pose.pose.pose.position.z;
            pose_z1 = -msg->detections.at(1).pose.pose.pose.position.y;

            origin.push_back(pose_x0);
            origin.push_back(pose_y0);
            origin.push_back(pose_z1);

            // ROS_INFO_STREAM("origin x: "<<origin[0]);
            // ROS_INFO_STREAM("origin y: "<<origin[1]);            
            // ROS_INFO_STREAM("origin z: "<<origin[2]);

        }

    }
    
}





int main(int argc, char** argv)
{   
    ros::init(argc,argv,"Pose");
    ros::NodeHandle nh;
    Tag tag(nh);

    ros::spin();

    return 0;   
}