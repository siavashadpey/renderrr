#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

#include "cuda_header.cuh"
#include "color.h"

class Image {
public:
    Image(int , int );

    ~Image();

    CUDA_CALLABLE void set_pixel(int, int, Color);
    CUDA_CALLABLE Color* pixel(int, int) const;

    CUDA_CALLABLE void dimensions(int&, int&) const;
    CUDA_CALLABLE Point pixel_location(int, int) const;

    int write_ppm(const char*) const;

protected:
	int height_;
    int width_;
    double dx_;
    double dy_;
    Color** pixels_; 

};

#endif