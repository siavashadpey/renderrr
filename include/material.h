#ifndef MATERIAL_H
#define MATERIAL_H

#include "cuda_header.cuh"
#include "color.h"

class Material {
public:
	Material(Color, float diffuse_coeff = 1.0f, float specular_coeff = 1.0f, float ambient_intensity = 0.05f, float reflection_intensity = 0.5f);
	CUDA_CALLABLE Material();
	CUDA_CALLABLE ~Material();
	CUDA_CALLABLE Color base_color() const;
	CUDA_CALLABLE float diffuse_coefficient() const;
	CUDA_CALLABLE float specular_coefficient() const;
	CUDA_CALLABLE float ambient_intensity() const;
	CUDA_CALLABLE float reflection_intensity() const;

protected:
	Color base_color_;
	float diffuse_coeff_;
	float specular_coeff_;
	float ambient_intensity_;
	float reflection_intensity_;
};


#endif
