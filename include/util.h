#ifndef UTILL_H
#define UTILL_H
#include <math.h>

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

#include "cuda_header.cuh"
#include "vector.h"



CUDA_CALLABLE float random_unit_float();
CUDA_CALLABLE Point random_point_on_unit_sphere();


#endif