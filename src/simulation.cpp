/// \file file
/// \brief This is the object motion simulation file

#include <ros/ros.h>
#include <ros/console.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/SetModelState.h>
#include <geometry_msgs/Pose.h>
#include <string>

#include <random>



// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2/LinearMath/Matrix3x3.h>

/// \brief generate random number
std::mt19937 & get_random()
{
    // static variables inside a function are created once and persist for the remainder of the program
    static std::random_device rd{}; 
    static std::mt19937 mt{rd()};
    // we return a reference to the pseudo-random number genrator object. This is always the
    // same object every time get_random is called
    
    return mt;
}


class Simulation
{
public:
    explicit Simulation(ros::NodeHandle& nh);


private:

    ros::NodeHandle nh_;

    ros::Subscriber pose_sub;

    ros::ServiceClient pose_srv;

    void PoseCallback(const gazebo_msgs::ModelStates::ConstPtr &msg);

    std::string name;

    geometry_msgs::Pose pose_;

    bool flag;

};


Simulation::Simulation(ros::NodeHandle& nh): nh_(nh)
{

    pose_sub = nh_.subscribe("gazebo/model_states",1,&Simulation::PoseCallback,this);

    pose_srv = nh_.serviceClient <gazebo_msgs::SetModelState>("gazebo/set_model_state");

    int freq = 10;

    ros::Rate rate(freq);



    while (ros::ok())
    {
        
        ROS_INFO("service started!");

        // Set max movement to 10 pixels distance (between -5 pixels ~ 5 pixels)
        double pixel2meter = 1.152e-3;  // in meters
        double min_move = -5 * pixel2meter, max_move = 5 * pixel2meter;

        // create random noise
        std::uniform_real_distribution<> d(min_move, max_move);
        std::uniform_real_distribution<> d2(0, max_move);           // Define a min bound range
        std::uniform_real_distribution<> d3(min_move, 0);           // Define a max bound range
        // Check if received pose callback
        if (flag)
        {
            // add noise to simulated model 
            gazebo_msgs::SetModelState set_model_state;
            
            set_model_state.request.model_state.model_name = name;
            set_model_state.request.model_state.pose.position.x = pose_.position.x;
            set_model_state.request.model_state.pose.position.y = pose_.position.y;
            set_model_state.request.model_state.pose.position.z = pose_.position.z;

            // Linear
            set_model_state.request.model_state.twist.linear.x = d(get_random());
            
            if (pose_.position.y <= -0.035)
            {
                set_model_state.request.model_state.twist.linear.y = d2(get_random());
            }
            else if (pose_.position.y >= 0.035)
            {
                set_model_state.request.model_state.twist.linear.y = d3(get_random()); 
            }
            else
            {
                set_model_state.request.model_state.twist.linear.y = d(get_random());
            }

            if (pose_.position.z <= 0.09)
            {
                set_model_state.request.model_state.twist.linear.z = d2(get_random());
            }
            else if (pose_.position.z >= 0.14)
            {
                set_model_state.request.model_state.twist.linear.z = d3(get_random());
            }
            else
            {
                set_model_state.request.model_state.twist.linear.z = d(get_random());
            }
            
            // Angular
            set_model_state.request.model_state.twist.angular.x = d(get_random());
            set_model_state.request.model_state.twist.angular.y = d(get_random());
            set_model_state.request.model_state.twist.angular.z = d(get_random());
        

            pose_srv.call(set_model_state);
        
            bool result = set_model_state.response.success;

            if(!result) ROS_WARN("Service failed!");

            else    ROS_INFO("Succeeded!");
        }


        flag = false;


        rate.sleep();

        ros::spinOnce();

    }
    
}


void Simulation::PoseCallback(const gazebo_msgs::ModelStates::ConstPtr & msg)
{
    name = msg->name.at(2);
    // ROS_INFO_STREAM("name: "<<name);

    pose_.position.x = msg->pose.at(2).position.x;
    pose_.position.y = msg->pose.at(2).position.y;
    pose_.position.z = msg->pose.at(2).position.z;
    
    // ROS_INFO_STREAM("pose y: "<<pose_.position.y);

    pose_.orientation.x = msg->pose.at(2).orientation.x;
    pose_.orientation.y = msg->pose.at(2).orientation.y;
    pose_.orientation.z = msg->pose.at(2).orientation.z;
    pose_.orientation.w = msg->pose.at(2).orientation.w;

    flag = true;
 
}




int main(int argc, char** argv)
{
    ros::init(argc,argv,"Simulation");
    ros::NodeHandle nh;
    Simulation simulation(nh);
    ros::spin();

    return 0;
}


/// end file