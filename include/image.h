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
    CUDA_CALLABLE Color pixel(int, int) const;

    CUDA_CALLABLE void dimensions(int&, int&) const;
    CUDA_CALLABLE Point pixel_location(int, int) const;

#ifdef __CUDACC__
    __host__ void cuda_malloc_memcpy_pointer_members(Image*);
    __host__ void cuda_memcpy_output();
    __host__ void cuda_free_pointer_members();
#endif

    int write_ppm(const char*) const;

protected:
    Color* pixels_;

	int height_;
    int width_;
    float dx_;
    float dy_;

#ifdef __CUDACC__
    Color* d_pixels_;
#endif
};

#endif
