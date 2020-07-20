#include <math.h>

#include "object.h"


Sphere::Sphere(float radius, Material* mat)
	: 	radius_(radius)
{
	material_ = mat;
}

Sphere::~Sphere()
{}

CUDA_CALLABLE float Sphere::radius() const
{
	return radius_;
}

CUDA_CALLABLE Color Sphere::base_color() const
{
	return material_->base_color();
}

CUDA_CALLABLE float Sphere::diffuse_coefficient() const
{
	return material_->diffuse_coefficient();
}

CUDA_CALLABLE float Sphere::specular_coefficient() const
{
	return material_->specular_coefficient();
}

CUDA_CALLABLE float Sphere::ambient_intensity() const
{
	return material_->ambient_intensity();
}

CUDA_CALLABLE float Sphere::reflection_intensity() const
{
	return material_->reflection_intensity();
}

CUDA_CALLABLE float Sphere::hit_distance(const Point center, const Ray& ray, Point& hit_location) const
{
	float dist = INFINITY;

	const Vector3d<float> camera_to_center = ray.origin() - center; 
	const Vector3d<float> ray_dir = ray.direction();
	const float a = 1.f;// ray_dir.dot(ray_dir) = 1
	const float b = 2.0f*ray_dir.dot(camera_to_center);
	const float c = camera_to_center.dot(camera_to_center) - radius_*radius_;

	const float disc = b*b - 4.0f*a*c;
	if (disc >= 0.f)
	{
		float t = (-b - sqrtf(disc))/(2.0f*a);
		// Positive t means object is in front of camera
		if (t > 0.f) { 
			hit_location = ray.direction()*t + ray.origin();
			dist = t;
		}
	}
	return dist;
}