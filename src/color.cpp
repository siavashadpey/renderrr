#include <stdio.h>

#include "vector.h"
#include "color.h"

Color::Color(double r, double g, double b)
	: Vector3d<double>(r, g, b)
{
}

Color::~Color() 
{
}

double* Color::RGB() const
{
	return this->values();
}

