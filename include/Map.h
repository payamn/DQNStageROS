#include "config.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Map
{
public:
    cv::Mat map_;

    cv::Mat getMap() { return map_; }
    int getWidth() { return width_; }
    int getHeight() { return height_; }
    
    void initMap(int width, int height);
    int setCell(double x, double y, unsigned int value);
protected:
    
    int width_;
    int height_;
};