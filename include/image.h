#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

#include "color.h"

class Image {
public:
    Image(int , int );

    ~Image();

    void set_pixel(int, int, Color);
    Color pixel(int, int) const;

    void dimensions(int&, int&) const;
    Point pixel_location(int, int) const;

    int write_ppm(const char*) const;

protected:
	int height_;
    int width_;
    double dx_;
    double dy_;
    std::vector<std::vector<Color> > pixels_;

};

#endif