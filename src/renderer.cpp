#include <cmath>

#include "renderer.h"

Renderer::Renderer(Scene* scene)
{
	scene_ = scene;
}

Renderer::~Renderer()
{}

void Renderer::render()
{

	int nrow;
	int ncol;
	scene_->image()->dimensions(nrow, ncol);

	// loop through each pixel in image
	/**
	 TODO: parallelize this. Ideally 1 pixel/thread
		Each thread needs access to the scene. Thus, this
		can be broadcasted.
		Threads are completely independent: each compute the
		color of the pixel that they own.
		Things to consider:
		1) Can memory coallescing be attained? even if Array Of Structures?
		2) What to put in shared memory?
		3) Can we achieve 1 pixel/thread or 
		   will we be memory-limited?
		4) Will thread divergence be an issue?
		   Not really. conditional statements 
		   are only if statements (i.e. no branching)
	**/
#ifdef __CUDA_ARCH__

#else
	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			Point pixel_loc = scene_->pixel_location(i, j);
			Vector3d<double> dir = scene_->camera_location().direction_to(pixel_loc);
			const Ray ray = Ray(scene_->camera_location(), dir);
			Color color = Color(0, 0, 0);
			color = trace_ray_(ray);
			set_pixel_color_(i, j, color);
		}
	}
#endif
}

CUDA_CALLABLE Color Renderer::trace_ray_(const Ray &ray, int emitted_ray_counter)
{
	Point hit_location;
	Vector3d<double> hit_normal_dir;
	Sphere* obj = scene_->object_hit(ray, hit_location, hit_normal_dir);

	Color color;
	if (is_hit_(obj)) {
		color_at_(obj, hit_location, hit_normal_dir, color);

		// trace ray reflected off of object
		if (emitted_ray_counter < max_rays_) {
			Point new_position = hit_location + hit_normal_dir*1.E-4;
			Vector3d<double> new_dir = ray.direction() - hit_normal_dir*hit_normal_dir.dot(ray.direction())*2.;
			new_dir.normalize();
			Ray reflected_ray = Ray(new_position, new_dir); 
			color += trace_ray_(reflected_ray, emitted_ray_counter+1)*obj->reflection_intensity();
		}
	}
	return color;
}

CUDA_CALLABLE bool Renderer::is_hit_(const Sphere* obj) const
{
	return obj;
}

CUDA_CALLABLE void Renderer::color_at_(const Sphere* obj, const Point& hit_location, const Vector3d<double>& hit_normal_dir, Color& color) const 
{
	// ambient color
	color += obj->base_color()*obj->ambient_intensity();

	Point hit_to_cam = scene_->camera_location() - hit_location;
	for (int i = 0; i < scene_->n_lights(); i++) {
		Light* current_light = scene_->light(i);

		// diffuse color - Lambert shading model
		Point hit_to_light = hit_location.direction_to(current_light->location());
		color += obj->base_color()*std::max(hit_to_light.dot(hit_normal_dir), 0.0)*obj->diffuse_coefficient();
		
		// specular color - Phong shading model
		Point R = hit_to_light + hit_to_cam;
		R.normalize();
		color += current_light->color()*obj->specular_coefficient()*pow(std::max(hit_normal_dir.dot(R), 0.0), 50.);
	}
}

CUDA_CALLABLE void Renderer::set_pixel_color_(int irow, int jcol, Color pixel_color)
{
	scene_->image()->set_pixel(irow, jcol, pixel_color);
}