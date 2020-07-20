#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"

class Material {
public:
	Material(Color, double diffuse_coeff = 1.0, double specular_coeff = 1.0, double ambient_intensity = 0.05, double reflection_intensity = 0.5);
	~Material();
	Color base_color() const;
	double diffuse_coefficient() const;
	double specular_coefficient() const;
	double ambient_intensity() const;
	double reflection_intensity() const;

protected:
	Color base_color_;
	double diffuse_coeff_;
	double specular_coeff_;
	double ambient_intensity_;
	double reflection_intensity_;
};


#endif