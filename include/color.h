#ifndef COLOR_H
#define COLOR_H

#include "cuda_header.cuh"
#include "vector.h"


class Color: public Vector3d<float> {
public:
    CUDA_CALLABLE Color(float r=0.f, float g=0.f, float b=0.f);

    CUDA_CALLABLE ~Color();

    CUDA_CALLABLE float red() const;
    CUDA_CALLABLE float green() const;
    CUDA_CALLABLE float blue() const;
    CUDA_CALLABLE Color& operator=(const Vector3d<float>&);

};

#endif
