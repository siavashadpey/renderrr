#include <cassert>
#include <math.h>

#include "ray.h"


Ray::Ray(Point origin, Vector3d<double> direction)
{

	assert(abs(direction.magnitude() - 1.) < 1E-5);
	origin_ = origin;
	direction_ = direction;
}

Ray::~Ray()
{}

Point Ray::origin() const
{
	return origin_;
}

Vector3d<double> Ray::direction() const
{
	return direction_;
}