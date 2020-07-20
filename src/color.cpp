#include <stdio.h>

#include "vector.h"
#include "color.h"

CUDA_CALLABLE Color::Color(double r, double g, double b)
	: Vector3d<double>(r, g, b)
{
}

CUDA_CALLABLE Color::~Color() 
{
}

CUDA_CALLABLE double* Color::RGB() const
{
	return this->values();
}

