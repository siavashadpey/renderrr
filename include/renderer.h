#ifndef RENDERER_H
#define RENDERER_H

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
	int emitted_ray_counter_ = 0; // this might cause issues when parallelizing
	const int max_rays_ = 50;

	Color trace_ray_(const Ray&);
	bool is_hit_(const Sphere*) const;
	void color_at_(const Sphere*, const Point&, const Vector3d<double>&, Color&) const;
	void set_pixel_color_(int, int, Color);
	void increment_emitted_ray_counter_();
	void reset_emitted_ray_counter_();
};

#endif