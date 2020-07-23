#include "light.h"

Light::Light(Point location, Color color)
{
	location_ = location;
	color_ = color;	
}

CUDA_CALLABLE Light::Light()
{}

CUDA_CALLABLE Light::~Light()
{}

CUDA_CALLABLE Point Light::location() const
{
	return location_;
}

CUDA_CALLABLE Color Light::color() const
{
	return color_;
}
