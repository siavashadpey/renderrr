#include <cassert>
#include <math.h>

#include "ray.h"


CUDA_CALLABLE Ray::Ray(Point origin, Vector3d<float> direction)
{

	//assert(abs(direction.magnitude() - 1.f) < .00001f);
	origin_ = origin;
	direction_ = direction;
}

CUDA_CALLABLE Ray::~Ray()
{}

CUDA_CALLABLE Point Ray::origin() const
{
	return origin_;
}

CUDA_CALLABLE Vector3d<float> Ray::direction() const
{
	return direction_;
}
