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

CUDA_CALLABLE float* Color::RGB() const
{
	return this->values();
}

