#include <stdio.h>

#include "vector.h"
#include "color.h"

CUDA_CALLABLE Color::Color(float r, float g, float b)
	: Vector3d<float>(r, g, b)
{
}

CUDA_CALLABLE Color::~Color() 
{
}

CUDA_CALLABLE float Color::red() const
{
	return this->vec_[0];
}

CUDA_CALLABLE float Color::green() const
{
	return this->vec_[1];
}


CUDA_CALLABLE float Color::blue() const
{
	return this->vec_[2];
}

CUDA_CALLABLE Color& Color::operator=(const Vector3d<float>& vec3d)
{
	Vector3d<float> vec3d_copy = vec3d;
	#pragma unroll (3)
	for (int i = 0; i < 3; i++) {
		vec_[i] = vec3d_copy[i];
	}
	return *this;
}
