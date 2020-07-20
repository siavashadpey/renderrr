#ifndef RAY_H
#define RAY_H

#include "cuda_header.cuh"
#include "vector.h"

class Ray {
public:
	CUDA_CALLABLE Ray(Point , Vector3d<double> );
	CUDA_CALLABLE ~Ray();

	CUDA_CALLABLE Point origin() const;
	CUDA_CALLABLE Vector3d<double> direction() const;

protected:
	Point origin_;
	Vector3d<double> direction_;
};

#endif