#ifndef COLOR
#define COLOR

#include "vector.h"

class Color: public Vector3d<double> {
public:
    Color(double r=0., double g=0., double b=0.);

    ~Color();

    double* RGB() const;

};

#endif