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


// useful mathematical functions
template<typename Number>
CUDA_CALLABLE Number Vector3d<Number>::magnitude() const {
	Number mag = (Number) 0.0f;
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		mag += vec_[i]*vec_[i];
	}
	return sqrtf(mag);
}

template<typename Number>
CUDA_CALLABLE void Vector3d<Number>::normalize() {
	Number mag = magnitude();
	#pragma unroll (3)
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
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		val += this->vec_[i]*other_v.vec_[i];
	}
	return val;
}

// operator overloading
template<typename Number>
CUDA_CALLABLE Vector3d<Number>& Vector3d<Number>::operator=(const Vector3d<Number>& other_v)
{
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		vec_[i] = other_v.vec_[i];
	}
	return *this;
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number>& Vector3d<Number>::operator+=(const Vector3d<Number>& other_v)
{
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		vec_[i] += other_v.vec_[i];
	}	
	return *this;
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator-() const
{
	Number val[3];
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		val[i] = -this->vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Number& Vector3d<Number>::operator[](int i) {
    return vec_[i];
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator+(const Vector3d<Number>& other_v) const 
{
	Number val[3];
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		val[i] = this->vec_[i] + other_v.vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator-(const Vector3d<Number>& other_v) const 
{
	Number val[3];
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		val[i] = this->vec_[i] - other_v.vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
}

template<typename Number>
CUDA_CALLABLE Vector3d<Number> Vector3d<Number>::operator*(const float& c) const
{
	Number val[3];
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		val[i] = c*this->vec_[i];
	}
	return Vector3d<Number>(val[0], val[1], val[2]);
} 

template class Vector3d<float>;
template class Vector3d<int>;
