#include "material.h"

Material::Material(Color base_color, bool is_reflective, float diffuse_coeff, float specular_coeff, float ambient_intensity, float reflection_intensity)
	:   is_reflective_(is_reflective),
		diffuse_coeff_(diffuse_coeff),
		specular_coeff_(specular_coeff),
		ambient_intensity_(ambient_intensity),
		reflection_intensity_(reflection_intensity)
{
	base_color_ = base_color;
}

CUDA_CALLABLE Material::Material()
{}

CUDA_CALLABLE Material::~Material()
{}

CUDA_CALLABLE Color Material::base_color() const
{
	return base_color_;
}

CUDA_CALLABLE float Material::diffuse_coefficient() const
{
	return diffuse_coeff_;
}

CUDA_CALLABLE float Material::specular_coefficient() const
{
	return specular_coeff_;
}

CUDA_CALLABLE float Material::ambient_intensity() const
{
	return ambient_intensity_;
}

CUDA_CALLABLE float Material::reflection_intensity() const
{
	return reflection_intensity_;
}

CUDA_CALLABLE bool Material::is_reflective() const
{
	return is_reflective_;
}
