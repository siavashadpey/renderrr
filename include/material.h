#ifndef MATERIAL_H
#define MATERIAL_H

#include "cuda_header.cuh"
#include "color.h"

class Material {
public:
	Material(Color, double diffuse_coeff = 1.0, double specular_coeff = 1.0, double ambient_intensity = 0.05, double reflection_intensity = 0.5);
	~Material();
	CUDA_CALLABLE Color base_color() const;
	CUDA_CALLABLE double diffuse_coefficient() const;
	CUDA_CALLABLE double specular_coefficient() const;
	CUDA_CALLABLE double ambient_intensity() const;
	CUDA_CALLABLE double reflection_intensity() const;

protected:
	Color base_color_;
	double diffuse_coeff_;
	double specular_coeff_;
	double ambient_intensity_;
	double reflection_intensity_;
};


#endif