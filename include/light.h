#ifndef LIGHT_H
#define LIGHT_H

#include "cuda_header.cuh"
#include "vector.h"
#include "color.h"

class Light {
public:
	Light(Point, Color);
	CUDA_CALLABLE Light();
	CUDA_CALLABLE ~Light();
	CUDA_CALLABLE Point location() const;
	CUDA_CALLABLE Color color() const;

protected:
	Point location_;
	Color color_;
};
#endif
