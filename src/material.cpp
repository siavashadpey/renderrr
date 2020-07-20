#include "material.h"

Material::Material(Color base_color, double diffuse_coeff, double specular_coeff, double ambient_intensity, double reflection_intensity)
	:	diffuse_coeff_(diffuse_coeff),
		specular_coeff_(specular_coeff),
		ambient_intensity_(ambient_intensity),
		reflection_intensity_(reflection_intensity)
{
	base_color_ = base_color;
}

Material::~Material()
{}

CUDA_CALLABLE Color Material::base_color() const
{
	return base_color_;
}

CUDA_CALLABLE double Material::diffuse_coefficient() const
{
	return diffuse_coeff_;
}

CUDA_CALLABLE double Material::specular_coefficient() const
{
	return specular_coeff_;
}

CUDA_CALLABLE double Material::ambient_intensity() const
{
	return ambient_intensity_;
}

CUDA_CALLABLE double Material::reflection_intensity() const
{
	return reflection_intensity_;
}