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

protected:
	Scene* scene_;
	const int max_rays_ = 50;

	CUDA_CALLABLE Color trace_ray_(const Ray&, int emitted_ray_counter = 1);
	CUDA_CALLABLE bool is_hit_(const Sphere*) const;
	CUDA_CALLABLE void color_at_(const Sphere*, const Point&, const Vector3d<float>&, Color&) const;
	CUDA_CALLABLE void set_pixel_color_(int, int, Color);
};

#endif