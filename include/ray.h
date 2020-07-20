#ifndef RAY_H
#define RAY_H

#include "vector.h"

class Ray {
public:
	Ray(Point , Vector3d<double> );
	~Ray();

	Point origin() const;
	Vector3d<double> direction() const;

protected:
	Point origin_;
	Vector3d<double> direction_;
};

#endif