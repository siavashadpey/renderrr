#include "util.h"

CUDA_CALLABLE float random_unit_float()
{
#ifdef __CUDA_ARCH__
	curandState state;
	int tId = blockDim.x*blockIdx.x + threadIdx.x 
	+ blockDim.y*blockIdx.y + threadIdx.y;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);
	return curand_uniform(&state);
#else
	return  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
#endif
//	return 1.f;
}

CUDA_CALLABLE Point random_point_on_unit_sphere() {
	float z = 2.f*random_unit_float() - 1.f; // z ~ U(-1, 1)
	float theta = 2.f*M_PI*random_unit_float(); // theta ~ U(0, 2*pi)
	float r = sqrtf(1.f - z*z);
	return Point(r*cosf(theta), r*sinf(theta), z);
}