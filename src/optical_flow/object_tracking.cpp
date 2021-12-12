#include <optical_flow/object_tracking.hpp>


void on_trackbar(int, void*){};

std::string intToString(int num)
{
    std::stringstream ss;
    ss<<num;
    return ss.str(); 
}


Tracking::Tracking(ros::NodeHandle &nh):nh_(nh)
{
    image_transport::ImageTransport it(nh_);

    // image subscriber
    img_sub = it.subscribe("/camera/color/image_raw",10,&Tracking::imageCallback,this);

    // depth subsriber
    depth_sub = nh_.subscribe("/camera/depth_registered/points",10,&Tracking::depthCallback,this);

    // position publisher
    pose_pub = nh_.advertise<optical_flow::position>("Position",10);


    // rate 
    ros::Rate loop_rate(10);

    optical_flow::position pose_msg;



    while (ros::ok())
    {
        // receive the filter object image
        if(flag)
        {
            int posX_1, posY_1;
            ROS_INFO("Position in X: %.4f",X_111);
            ROS_INFO("Position in Y: %.4f",Y_111);
            ROS_INFO("Position in Z: %.4f",Z_111);

            posX_1 = pos_X;
            posY_1 = pos_Y;

            pose_msg.Pose_XYZ.clear();
            
            pose_msg.centroid_pixel_x = posX_1;
            pose_msg.centroid_pixel_y = posY_1;


            geometry_msgs::Point position_XYZ;
            // int max_range = 1000;
            
            // if (X_111<max_range && Y_111<max_range && Z_111<max_range)
            // {
            position_XYZ.x = X_111;
            position_XYZ.y = Y_111;
            position_XYZ.z = Z_111;
            pose_msg.Pose_XYZ.push_back(position_XYZ);
            pose_pub.publish(pose_msg);
            // }

            // loop_rate.sleep();

        }
        else
        {
            pose_msg.Pose_XYZ.clear();
            pose_msg.centroid_pixel_x = 0;
            pose_msg.centroid_pixel_y = 0;


            geometry_msgs::Point position_XYZ;
            position_XYZ.x = 0;
            position_XYZ.y = 0;
            position_XYZ.z = 0;
            pose_msg.Pose_XYZ.push_back(position_XYZ);

            pose_pub.publish(pose_msg);

            // loop_rate.sleep();
        }

        loop_rate.sleep();

        ros::spinOnce();

    }
    

}




void Tracking::createTrackBars()
{
    // create windown for track bar
    cv::namedWindow("Track_bars",0);
    // char TrackbarName[50];
    cv::createTrackbar("H_MIN","Track_bars",&H_MIN,H_MAX,on_trackbar);
    cv::createTrackbar("H_MAX","Track_bars",&H_MAX,H_MAX,on_trackbar);
    cv::createTrackbar("S_MIN","Track_bars",&S_MIN,S_MAX,on_trackbar);
    cv::createTrackbar("S_MAX","Track_bars",&S_MAX,S_MAX,on_trackbar);
    cv::createTrackbar("V_MIN","Track_bars",&V_MIN,V_MAX,on_trackbar);
    cv::createTrackbar("V_MAX","Track_bars",&V_MAX,V_MAX,on_trackbar);

}

void Tracking::drawObject(int x, int y, cv::Mat &frame)
{
    cv::circle(frame, cv::Point(x,y), 20, cv::Scalar(0,255,0),2);

    // if(y-25>0)
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x,y-25),cv::Scalar(0,255,0),2);
    // }
    // else
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x,0),cv::Scalar(0,255,0),2);
    // }
        
    // if(y+25 < FRAME_HEIGHT)
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x,y+25),cv::Scalar(0,255,0),2);
    // }
    // else
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x,FRAME_HEIGHT),cv::Scalar(0,255,0),2);
    // }

    // if(x-25>0)
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x-25,y),cv::Scalar(0,255,0),2);
    // }
    // else
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(0,y),cv::Scalar(0,255,0),2);
    // }
        
    // if(x+25 < FRAME_WIDTH)
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(x+25,y),cv::Scalar(0,255,0),2);
    // }
    // else
    // {
    //     cv::line(frame,cv::Point(x,y),cv::Point(FRAME_WIDTH,y),cv::Scalar(0,255,0),2);
    // }

    cv::putText(frame,"Pixel x: "+intToString(x) +"," + "Pixel y: " + intToString(y),cv::Point(20,150),1,2,cv::Scalar(0,255,0),2);
    pos_X = x;
    pos_Y = y;

    cv::putText(frame,"X Y Z coordinates:",cv::Point(20,200),1,2,cv::Scalar(0,255,0),2);
    cv::putText(frame,"X = " + intToString(x_pos) + "mm",cv::Point(20,250),1,2,cv::Scalar(0,255,0),2);
    cv::putText(frame,"Y = " + intToString(y_pos) + "mm",cv::Point(20,300),1,2,cv::Scalar(0,255,0),2);
    cv::putText(frame,"Z = " + intToString(z_pos) + "mm",cv::Point(20,350),1,2,cv::Scalar(0,255,0),2);


}

void Tracking::morphOps(cv::Mat &thresh)
{
    cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(8,8));

    cv::erode(thresh, thresh, erodeElement);
    cv::erode(thresh, thresh, erodeElement);

    cv::dilate(thresh, thresh, dilateElement);
    cv::dilate(thresh, thresh, dilateElement);


}


void Tracking::trackFilterObject(int &x, int &y, cv::Mat threshold, cv::Mat &cameraFeed)
{

    cv::Mat temp;
    threshold.copyTo(temp);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // find contours of the object
    cv::findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);

    // use moment to find the centroid
    double ref_Area = 0;
    bool found = false;
    // int l = hierarchy.size();
    // ROS_INFO_STREAM("hierarchy: "<<l);
    if(hierarchy.size() > 0)
    {
        int numObjects = hierarchy.size();
        if (numObjects < MAX_NUM_OBJECTS)
        {
            for (int i=0; i>=0;i = hierarchy[i][0])
            {
                cv::Moments mu = cv::moments((cv::Mat)contours[i]);
                double area = mu.m00;
                ROS_INFO_STREAM("area: "<<area);

                // keep the largest area of interest in the image
                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>ref_Area)
                {
                    // find centroid
                    x = mu.m10 / area;
                    ROS_INFO_STREAM("x:"<<x);
                    y = mu.m01 / area;
                    ROS_INFO_STREAM("y:"<<y);
                    found = true;
                    flag = true;
                    ref_Area = area;
                }
                else
                {
                    found = false;
                    flag = false;
                }
            }
        }
        
        if (found)
        {
            cv::putText(cameraFeed, "Position object tracking",cv::Point(0,50),2,1,cv::Scalar(0,255,0),2);

            drawObject(x,y,cameraFeed);
        }

        else
        {
            cv::putText(cameraFeed,"To much noise, adjust the noise filter!", cv::Point(0,50),1,2,cv::Scalar(0,255,0),2);
        }

    }



}

void Tracking::getXYZ(int x, int y)
{
    int array_Pos = y * my_pcl.row_step + x * my_pcl.point_step;
    int array_PosX = array_Pos + my_pcl.fields[0].offset;       // X has offset of 0
    int array_PosY = array_Pos + my_pcl.fields[1].offset;       // Y has offset of 4
    int array_PosZ = array_Pos + my_pcl.fields[2].offset;       // Z has offset of 8
    
    float X = 0.0;
    float Y = 0.0;
    float Z = 0.0;

    memcpy(&X, &my_pcl.data[array_PosX],sizeof(float));
    memcpy(&Y, &my_pcl.data[array_PosY],sizeof(float));
    memcpy(&Z, &my_pcl.data[array_PosZ],sizeof(float));

    // position in meters
    if (!isnan(X) && !isnan(Y) && !isnan(Z))
    {
        X_111 = X;
        Y_111 = Y;
        Z_111 = Z;

        // position in mm
        x_pos = int(X*1000);
        y_pos = int(Y*1000);
        z_pos = int(Z*1000);
    }


}

void Tracking::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // ROS_INFO("Hello");
    // cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exeception: %s",e.what());
        return;
    }

    bool trackObjects = true;
    bool useMorphOps = true;

    cv::Mat HSV;
    cv::Mat threshold;
    int x=0, y=0;

    createTrackBars();

    std::cout<<"The object tracking"<<std::endl;

    cv::cvtColor(cv_ptr->image, HSV,cv::COLOR_BGR2HSV);

    cv::inRange(HSV,cv::Scalar(H_MIN,S_MIN,V_MIN),cv::Scalar(H_MAX,S_MAX,V_MAX), threshold);

    if (useMorphOps)
    {
        morphOps(threshold);
    }

    if (trackObjects)
    {
        trackFilterObject(x,y,threshold,cv_ptr->image);

    }
    // cv::Mat res;
    // cv::bitwise_and(cv_ptr->image,threshold,res);
    // cv::imshow(window3,res);

    //show frames
    cv::imshow(window2,threshold);
    //show camera image
    cv::imshow(OPENCV_WINDOW,cv_ptr->image);

    cv::waitKey(1);

}

void Tracking::depthCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    ROS_INFO("Hello");
    my_pcl = *msg;
    // ROS_INFO_STREAM("my pcl"<<my_pcl.data[0]);

    getXYZ(pos_X,pos_Y);

}




