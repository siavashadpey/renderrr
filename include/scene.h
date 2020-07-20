#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "vector.h"
#include "object.h"
#include "image.h"
#include "ray.h"
#include "light.h"

class Scene {
public:
	Scene(Point);
	~Scene();
	
	void add_object(Sphere*, Point*);
	void add_light(Light*);

	// member interface
	Image* image() const;
	Point pixel_location(int, int) const;
	Point camera_location() const;
	int n_objects() const;
	int n_lights() const;
	Light* light(int) const;

	// helper methods
	Sphere* object_hit(const Ray&, Point&, Vector3d<double>&) const;

protected:
	Point camera_location_;

	Image* image_;
	Point image_bl_location_; // Image's bottom left corner location
	double image_dims_[2];

	std::vector<Sphere *> objects_;
	std::vector<Point* > object_locations_;

	std::vector<Light *> lights_;


};

#endif