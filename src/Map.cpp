#include "Map.h"
#include "ros/ros.h"
#include <iostream>
#include <cmath>
#include <limits>

void Map::initMap(int width, int height, Stg::ModelPosition* robot)
{
    width_ = round(width / (float) MAP_LEAST_COUNT);
    height_ = round(height / (float) MAP_LEAST_COUNT);
    map_ = cv::Mat::zeros(height_, width_, CV_8UC1);

    robot_pose_ = robot->GetPose();
    robot_size_ = robot->GetGeom().size;
    
    robot_grid_size_x_ = round(robot_size_.x / (float) MAP_LEAST_COUNT);
    robot_grid_size_y_ = round(robot_size_.y / (float) MAP_LEAST_COUNT);
        
    drawRobot(CURRENT_ROBOT_COLOR);
}

int Map::drawRobot(unsigned char color)
{
    int grid_x;
    int grid_y;

    if (convertToGridCoords(robot_pose_.x, robot_pose_.y, grid_x, grid_y)) {
        return 1;
    }

    cv::RotatedRect robot_rect = cv::RotatedRect(
        cv::Point2f(grid_x, grid_y), 
        cv::Size2f(robot_grid_size_x_, robot_grid_size_y_), 
        -robot_pose_.a * 180 / M_PI
    );

    cv::Point2f vertices2f[4];
    cv::Point vertices[4];
    
    robot_rect.points(vertices2f);

    for(int i = 0; i < 4; ++i) {
        vertices[i] = vertices2f[i];
    }
    
    cv::fillConvexPoly(
        map_,
        vertices,
        4,
        color
    );


    return 0;
}

int Map::updateMap(Stg::Pose new_robot_pose, const Stg::ModelRanger::Sensor& sensor)
{
    if (updateRobotPose(new_robot_pose)) {
        return 1;
    }

    if (updateLaserScan(sensor)) {
        return 1;
    }

    return 0;
}


int Map::updateRobotPose(Stg::Pose new_robot_pose)
{
    drawRobot(PREVIOUS_ROBOT_TRAJECTORY_COLOR);
    // setCell(robot_pose_.x, robot_pose_.y, PREVIOUS_ROBOT_TRAJECTORY_COLOR);   

    robot_pose_ = new_robot_pose;
    return drawRobot(CURRENT_ROBOT_COLOR);
    // return setCell(robot_pose_.x, robot_pose_.y, CURRENT_ROBOT_COLOR);
}

int Map::updateLaserScan(const Stg::ModelRanger::Sensor& sensor)
{
    // get the data
    const std::vector<Stg::meters_t>& scan = sensor.ranges;
  
    uint32_t sample_count = scan.size();
    if( sample_count < 1 )
        return 1;

    double laser_orientation = robot_pose_.a - sensor.fov/2.0;
    double angle_increment = sensor.fov/(double)(sensor.sample_count-1);

    for (uint32_t i = 0; i < sample_count; i++) {
        // normalize the angle
        laser_orientation = atan2(sin(laser_orientation), cos(laser_orientation));
        
        double laser_x, laser_y;
        laser_x = robot_pose_.x +  scan[i] * cos(laser_orientation);
        laser_y = robot_pose_.y +  scan[i] * sin(laser_orientation);

        drawLine(robot_pose_.x, robot_pose_.y, laser_x, laser_y);

        if ( scan[i] < (sensor.range.max - std::numeric_limits<float>::min()) ) {
            // ROS_INFO("laser: %f, %f", laser_x, laser_y);

            // draw obstacle
            setCell(
                laser_x,
                laser_y,
                OBSTACLE_COLOR
            );

        }

        laser_orientation += angle_increment;
    }

    return 0;
}

int Map::setCell(double x, double y, unsigned int value)
{
    int grid_x;
    int grid_y;

    if (convertToGridCoords(x, y, grid_x, grid_y)) {
        return 1;
    }

    // mat.at<uchar>(row, column)
    map_.at<uint8_t>(grid_y, grid_x) = value;

    return 0;
}

/********* Utilities ***************/

/*
 @brief convert coords from continuous world coordinate to discrete image coord
*/
int Map::convertToGridCoords(double x, double y, int &grid_x, int &grid_y)
{
    grid_x = round( x/(float) MAP_LEAST_COUNT + width_/ 2 );
    grid_y = round( y/(float) MAP_LEAST_COUNT + height_/2 );

    if (
        grid_x < 0 ||
        grid_y < 0 ||
        grid_x  > width_ ||
        grid_y > height_
    ) {
        return 1;
    } else {
        return 0;
    }
}

int Map::drawLine(double x1, double y1, double x2, double y2)
{

    int grid_x1, grid_y1, grid_x2, grid_y2;

    if ( convertToGridCoords(x1, y1, grid_x1, grid_y1) ) {
        return 1;
    }

    if ( convertToGridCoords(x2, y2, grid_x2, grid_y2) ) {
        return 1;
    }

    cv::LineIterator it(map_, cv::Point(grid_x1, grid_y1), cv::Point(grid_x2, grid_y2));
    
    for(int i = 0; i < it.count; i++, ++it)
    {
        if (map_.at<uint8_t>(it.pos()) != PREVIOUS_ROBOT_TRAJECTORY_COLOR) {
            map_.at<uint8_t>(it.pos()) = EXPLORED_AREA_COLOR;
        }
    }
    
    return 0;

}
