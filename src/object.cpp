#include <math.h>

#include "object.h"


Sphere::Sphere(double radius, Material* mat)
	: 	radius_(radius)
{
	material_ = mat;
}

Sphere::~Sphere()
{}

double Sphere::radius() const
{
	return radius_;
}

Color Sphere::base_color() const
{
	return material_->base_color();
}

double Sphere::diffuse_coefficient() const
{
	return material_->diffuse_coefficient();
}

double Sphere::specular_coefficient() const
{
	return material_->specular_coefficient();
}

double Sphere::ambient_intensity() const
{
	return material_->ambient_intensity();
}

double Sphere::reflection_intensity() const
{
	return material_->reflection_intensity();
}

double Sphere::hit_distance(const Point center, const Ray& ray, Point& hit_location) const
{
	double dist = INFINITY;

	const Vector3d<double> camera_to_center = ray.origin() - center; 
	const Vector3d<double> ray_dir = ray.direction();
	const double a = 1.;// ray_dir.dot(ray_dir) = 1
	const double b = 2.0*ray_dir.dot(camera_to_center);
	const double c = camera_to_center.dot(camera_to_center) - radius_*radius_;

	const double disc = b*b - 4.0*a*c;
	if (disc >= 0)
	{
		double t = (-b - sqrt(disc))/(2.0*a);
		// Positive t means object is in front of camera
		if (t > 0) { 
			hit_location = ray.direction()*t + ray.origin();
			dist = t;
		}
	}
	return dist;
}