/// \file object_tracking.hpp
/// \brief  header file for object_tracking
#ifndef OBJECT_TRACKING_INCLUDE_GUARD_HPP
#define OBJECT_TRACKING_INCLUDE_GUARD_HPP

#include <ros/ros.h>
#include <ros/console.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <string>
#include <vector>
#include <sstream>
#include <optical_flow/position.h>

// Global parameters 


// viewer variables
static const std::string OPENCV_WINDOW = "Image Window";
static const std::string window1 = "HSV Image";
static const std::string window2 = "Thresholded Image";
static const std::string window3 = "Combined Image";
static const std::string window4 = "After Morphological Operations";
static const std::string Trackbar = "Track bars";


// track_bar func
void on_trackbar(int, void*);

/// \brief  convert int to string 
/// \param  num  integer number
/// \return string num
std::string intToString(int num);


/// \brief Obeject_tracking class
class Tracking
{

public:
    /// \brief default constructor for the class
    explicit Tracking(ros::NodeHandle &nh);


private:

    // HSV color spaces range
    int H_MIN = 0;
    int H_MAX = 256;
    int S_MIN = 0;
    int S_MAX = 256;
    int V_MIN = 0;
    int V_MAX = 256;

    // flag to check if object is detected
    bool flag;  

    // pixels x, y
    int pos_X = 0;
    int pos_Y = 0;

    // X,Y,Z distance to camera depth frame
    float X_111 = 0.0;
    float Y_111 = 0.0;
    float Z_111 = 0.0;
    
    // x , y and z position
    int x_pos, y_pos, z_pos;

    // filter parameters
    static const int FRAME_WIDTH = 640;
    static const int FRAME_HEIGHT = 480;
    static const int MAX_NUM_OBJECTS = 50;
    static const int MIN_OBJECT_AREA = 20 * 20;
    static const int MAX_OBJECT_AREA = FRAME_WIDTH * FRAME_HEIGHT / 1.5;



    // private nodehandle
    ros::NodeHandle nh_;
    
    // color image subscriber
    image_transport::Subscriber img_sub;

    // depth image subscriber
    ros::Subscriber depth_sub;

    // position msg publisher
    ros::Publisher pose_pub;

    //  conversion between opencv and ros
    cv_bridge::CvImagePtr cv_ptr;

    // pointcloud
    sensor_msgs::PointCloud2 my_pcl;

    /// \brief func to create HSV Trackbar
    void createTrackBars();

    /// \brief func to draw the center of objects
    /// \param x  center pixel x 
    /// \param y  center pixel y
    /// \param frame  src image input
    /// \returns none
    void drawObject(int x, int y, cv::Mat &frame);

    /// \brief func to filter the image
    /// \param thresh src image (HSV)
    /// \returns none 
    void morphOps(cv::Mat &thresh);

    /// \brief func to track the filtered object
    /// \param x x coordinate of center pixel     
    /// \param y y coordinate of center pixel
    /// \param threshold HSV filtered image
    /// \param cameraFeed original camera image
    /// \returns none 
    void trackFilterObject(int &x, int &y, cv::Mat threshold, cv::Mat &cameraFeed);

    /// \brief func to calculate X,Y and Z location from objects centroid(x,y pixel) to camera depth frame
    /// \param x x pixel coordinate
    /// \param y y pixel coordinate
    /// \returns none
    void getXYZ(int x, int y);

    /// \brief color image callback
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

    /// \brief depth image callback
    void depthCallback(const sensor_msgs::PointCloud2ConstPtr &msg);




};













#endif

/// end file