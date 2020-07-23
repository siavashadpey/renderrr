#ifndef RENDERER_H
#define RENDERER_H

#include "cuda_header.cuh"
#include "scene.h"
#include "object.h"
#include "ray.h"

class Renderer {
public:
	Renderer(Scene*);
	~Renderer();
	void render();
	CUDA_CALLABLE Scene* scene() const;
	CUDA_CALLABLE Color trace_ray(Ray&);
	CUDA_CALLABLE void set_pixel_color(int, int, Color);
	CUDA_CALLABLE int max_rays() const;

protected:
	Scene* scene_;
	
	const int max_rays_ = 50;

	CUDA_CALLABLE void color_at_(const Sphere&, const Point&, const Vector3d<float>&, Color&) const;
};

#endif

#ifdef __CUDACC__
__global__ void set_pixel_color_kernel(Renderer*, int, int);

#define NTHREADS 16
#endif
