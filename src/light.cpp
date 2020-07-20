#include "light.h"

Light::Light(Point location, Color color)
{
	location_ = location;
	color_ = color;	
}

Light::~Light()
{}

Point Light::location() const
{
	return location_;
}

Color Light::color() const
{
	return color_;
}