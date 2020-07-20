#ifndef SPHERE_H
#define SPHERE_H

#include "vector.h"
#include "material.h"
#include "color.h"
#include "ray.h"

class Sphere {
public:
	Sphere(double, Material*);
	~Sphere();
	Color base_color() const;
	double radius() const;
	double diffuse_coefficient() const;
	double specular_coefficient() const;
	double ambient_intensity() const;
	double reflection_intensity() const;

	double hit_distance(const Point, const Ray&, Point&) const;

protected:
	double radius_;
	Material* material_;
};

#endif