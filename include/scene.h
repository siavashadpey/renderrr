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
	Scene(Point, int, int);
	~Scene();
	
	void add_object(Sphere, Point);
	void add_light(Light);

	// member interface
	CUDA_CALLABLE Image* image() const;
	CUDA_CALLABLE Point pixel_location(int, int) const;
	CUDA_CALLABLE Point camera_location() const;
	CUDA_CALLABLE void set_n_objects(int) const;
	CUDA_CALLABLE int n_objects() const;
	CUDA_CALLABLE int n_lights() const;
	CUDA_CALLABLE void set_n_lights(int) const;
	CUDA_CALLABLE Light light(int) const;

	// helper methods
	CUDA_CALLABLE Sphere object_hit(const Ray&, Point&, Vector3d<float>&, bool&) const;
	Image* image_;
	Point* object_locations_;
	Sphere* objects_;
	Light* lights_;
protected:
	Point camera_location_;

	
	Point image_bl_location_; // Image's bottom left corner location
	float image_dims_[2];

	int n_objects_;
	int i_object_;


	int n_lights_;
	int i_light_;
	


};

#endif
