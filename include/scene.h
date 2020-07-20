#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "cuda_header.cuh"
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
	CUDA_CALLABLE Image* image() const;
	CUDA_CALLABLE Point pixel_location(int, int) const;
	CUDA_CALLABLE Point camera_location() const;
	CUDA_CALLABLE int n_objects() const;
	CUDA_CALLABLE int n_lights() const;
	CUDA_CALLABLE Light* light(int) const;

	// helper methods
	CUDA_CALLABLE Sphere* object_hit(const Ray&, Point&, Vector3d<float>&) const;

protected:
	Point camera_location_;

	Image* image_;
	Point image_bl_location_; // Image's bottom left corner location
	float image_dims_[2];

	std::vector<Sphere *> objects_;
	std::vector<Point* > object_locations_;

	std::vector<Light *> lights_;


};

#endif