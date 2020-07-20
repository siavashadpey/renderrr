#ifndef COLOR
#define COLOR

#include "cuda_header.cuh"
#include "vector.h"


class Color: public Vector3d<float> {
public:
    CUDA_CALLABLE Color(float r=0.f, float g=0.f, float b=0.f);

    CUDA_CALLABLE ~Color();

    CUDA_CALLABLE float* RGB() const;

};

#endif