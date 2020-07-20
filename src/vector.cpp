#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "vector.h"

template<typename Number>
CUDA_CALLABLE Vector3d<Number>::Vector3d(Number x, Number y, Number z) {
    vec_[0] = x;
    vec_[1] = y;
    vec_[2] = z;
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number>::Vector3d() 
{
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number>::~Vector3d() {
}

template<typename Number>
CUDA_CALLABLE Number* Vector3d<Number>::values() const {
    return vec_;
}

// useful mathematical functions
template<typename Number>
CUDA_CALLABLE Number Vector3d<Number>::magnitude() const {
	Number mag = (Number) 0.0;
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		mag += vec_[i]*vec_[i];
	}
	return sqrt(mag);
}

template<typename Number>
CUDA_CALLABLE void Vector3d<Number>::normalize() {
	Number mag = magnitude();
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		vec_[i] /=mag;
	}
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::direction_to(const Vector3d<Number>& other_v) const
{
	Vector3d<Number> uv = other_v - *this;
	uv.normalize();
	return uv;
}

template<typename Number>
CUDA_CALLABLE Number Vector3d<Number>::dot(const Vector3d<Number>& other_v) const
{
	Number val = (Number) 0.0;
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		val += this->vec_[i]*other_v.values()[i];
	}
	return val;
}

// operator overloading
template<typename Number>
CUDA_CALLABLE Vector3d<Number>& Vector3d<Number>::operator=(const Vector3d<Number>& other_v)
{
	vec_ = other_v.vec_;
	return *this;
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number>& Vector3d<Number>::operator+=(const Vector3d<Number>& other_v)
{
	vec_ = (*this + other_v).values();
	return *this;
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator-() const
{
	Number* val = (Number*) malloc(3*sizeof(Number));

	#pragma unroll
	for (int i = 0; i < 3; i++) {
		val[i] = -this->vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator+(const Vector3d<Number>& other_v) const 
{
	Number* val = (Number*) malloc(3*sizeof(Number));

	#pragma unroll
	for (int i = 0; i < 3; i++) {
		val[i] = this->vec_[i] + other_v.values()[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator-(const Vector3d<Number>& other_v) const 
{
	Number* val = (Number*) malloc(3*sizeof(Number));

	#pragma unroll
	for (int i = 0; i < 3; i++) {
		val[i] = this->vec_[i] - other_v.values()[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator*(const double& c) const
{
	Number* val = (Number*) malloc(3*sizeof(Number));

	#pragma unroll
	for (int i = 0; i < 3; i++) {
		val[i] = c*this->vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
} 

template class Vector3d<double>;
template class Vector3d<int>;