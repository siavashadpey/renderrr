#ifndef SPHERE_H
#define SPHERE_H

#include "cuda_header.cuh"
#include "vector.h"
#include "material.h"
#include "color.h"
#include "ray.h"

class Sphere {
public:
	Sphere(float, Material);
	CUDA_CALLABLE Sphere();
	CUDA_CALLABLE ~Sphere();
	CUDA_CALLABLE Color base_color() const;
	CUDA_CALLABLE float radius() const;
	CUDA_CALLABLE float diffuse_coefficient() const;
	CUDA_CALLABLE float specular_coefficient() const;
	CUDA_CALLABLE float ambient_intensity() const;
	CUDA_CALLABLE float reflection_intensity() const;

	CUDA_CALLABLE float hit_distance(const Point, const Ray&, Point&) const;

protected:
	float radius_;
	Material material_;
};

#endif
