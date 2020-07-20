#ifndef COLOR
#define COLOR

#include "cuda_header.cuh"
#include "vector.h"


class Color: public Vector3d<double> {
public:
    CUDA_CALLABLE Color(double r=0., double g=0., double b=0.);

    CUDA_CALLABLE ~Color();

    CUDA_CALLABLE double* RGB() const;

};

#endif