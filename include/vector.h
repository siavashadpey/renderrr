#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <stdlib.h>

#include "cuda_header.cuh"

template<typename Number>
class Vector3d {
public:
    CUDA_CALLABLE Vector3d(Number , Number , Number );
    CUDA_CALLABLE Vector3d();

    CUDA_CALLABLE ~Vector3d();

    CUDA_CALLABLE Number* values() const;
    CUDA_CALLABLE Number magnitude() const;
    CUDA_CALLABLE void normalize();
    CUDA_CALLABLE Vector3d<Number> direction_to(const Vector3d<Number>&) const;
    CUDA_CALLABLE Number dot(const Vector3d<Number>&) const;

    CUDA_CALLABLE Vector3d<Number>& operator=(const Vector3d<Number>&);
    CUDA_CALLABLE Vector3d<Number>& operator+=(const Vector3d<Number>&);
    CUDA_CALLABLE Vector3d<Number> operator-() const;
    CUDA_CALLABLE Vector3d<Number> operator+(const Vector3d<Number>&) const;
    CUDA_CALLABLE Vector3d<Number> operator-(const Vector3d<Number>&) const;
    CUDA_CALLABLE Vector3d<Number> operator*(const float&) const;

protected:
	Number* vec_ = (Number*) malloc(3*sizeof(Number));

};

using Point = Vector3d<float>;

#endif