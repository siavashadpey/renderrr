#ifndef RAY_H
#define RAY_H

#include "cuda_header.cuh"
#include "vector.h"

class Ray {
public:
	CUDA_CALLABLE Ray(Point , Vector3d<float>);
	CUDA_CALLABLE ~Ray();

	CUDA_CALLABLE Point origin() const;
	CUDA_CALLABLE Vector3d<float> direction() const;
	CUDA_CALLABLE Ray& operator=(const Ray&);

protected:
	Point origin_;
	Vector3d<float> direction_;
};

#endif
