#include "Map.h"
#include <iostream>
#include <cmath>

void Map::initMap(int width, int height)
{
    width_ = width / (float) MAP_LEAST_COUNT;
    height_ = height / (float) MAP_LEAST_COUNT;

    map_ = cv::Mat::zeros(height_, width_, CV_8UC1);

}

int Map::setCell(double x, double y, unsigned int value)
{
    int grid_x = round( x/(float) MAP_LEAST_COUNT + width_/ 2 );
    int grid_y = round( y/(float) MAP_LEAST_COUNT + height_/2 );

    if (
        grid_x < 0 ||
        grid_y < 0 ||
        grid_x  > width_ ||
        grid_y > height_
    ) {
        return 1;
    }

    // mat.at<uchar>(row, column)
    map_.at<uint8_t>(grid_y, grid_x) = value;

    return 0;
}