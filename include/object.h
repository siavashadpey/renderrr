#ifndef SPHERE_H
#define SPHERE_H

#include "cuda_header.cuh"
#include "vector.h"
#include "material.h"
#include "color.h"
#include "ray.h"

class Sphere {
public:
	Sphere(double, Material*);
	~Sphere();
	CUDA_CALLABLE Color base_color() const;
	CUDA_CALLABLE double radius() const;
	CUDA_CALLABLE double diffuse_coefficient() const;
	CUDA_CALLABLE double specular_coefficient() const;
	CUDA_CALLABLE double ambient_intensity() const;
	CUDA_CALLABLE double reflection_intensity() const;

	CUDA_CALLABLE double hit_distance(const Point, const Ray&, Point&) const;

protected:
	double radius_;
	Material* material_;
};

#endif