#include "config.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stage.hh>

class Map
{
public:
    cv::Mat map_;

    cv::Mat getMap() { return map_; }
    int getWidth() { return width_; }
    int getHeight() { return height_; }

    void initMap(int width, int height, Stg::Pose robot_pose);
    
    // @output: 0 - success, 1 - failure
    int updateMap(Stg::Pose new_robot_pose, const Stg::ModelRanger::Sensor& sensor);
    int updateRobotPose(Stg::Pose new_robot_pose);
    int updateLaserScan(const Stg::ModelRanger::Sensor& sensor);

    // utilities
    int convertToGridCoords(double x, double y, int &grid_x, int &grid_y);
    int drawLine(double x1, double y1, double x2, double y2);
    
protected:
    
    int width_;
    int height_;
    Stg::Pose robot_pose_;

    int setCell(double x, double y, unsigned int value);
    
};