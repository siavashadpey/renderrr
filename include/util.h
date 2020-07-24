#include <math.h>

#include "vector.h"

 inline float random_float() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

inline Point random_point_on_unit_sphere()
{
	float z = 2.f*random_float() - 1.f; // z ~ U(-1, 1)
	float theta = 2.0f*M_PI*random_float(); // theta ~ U(0, 2*pi)
	float r = sqrtf(1.0f - z*z);
	return Point(r*cosf(theta), r*sinf(theta), z);
}

