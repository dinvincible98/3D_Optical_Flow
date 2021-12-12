#include <optical_flow/object_tracking.hpp>


int main(int argc,char** argv)
{
    ros::init(argc,argv,"object_tracking");
    ros::NodeHandle nh;
    Tracking tracking(nh);
    ros::spin();
    return 0;
}