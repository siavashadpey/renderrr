#ifndef LIGHT_H
#define LIGHT_H

#include "vector.h"
#include "color.h"

class Light {
public:
	Light(Point, Color);
	~Light();
	Point location() const;
	Color color() const;

protected:
	Point location_;
	Color color_;
};
#endif