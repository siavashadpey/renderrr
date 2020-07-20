#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <stdlib.h>

template<typename Number>
class Vector3d {
public:
    Vector3d(Number , Number , Number );
    Vector3d();

    ~Vector3d();

    Number* values() const;
    Number magnitude() const;
    void normalize();
    Vector3d<Number> direction_to(const Vector3d<Number>&) const;
    Number dot(const Vector3d<Number>&) const;

    Vector3d<Number>& operator=(const Vector3d<Number>&);
    Vector3d<Number>& operator+=(const Vector3d<Number>&);
    Vector3d<Number> operator-() const;
    Vector3d<Number> operator+(const Vector3d<Number>&) const;
    Vector3d<Number> operator-(const Vector3d<Number>&) const;
    Vector3d<Number> operator*(const double&) const;

protected:
	Number* vec_ = (Number*) malloc(3*sizeof(Number));

};

using Point = Vector3d<double>;

#endif