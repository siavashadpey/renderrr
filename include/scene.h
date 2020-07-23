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
	CUDA_CALLABLE Sphere* objects() const;

	// helper methods
	CUDA_CALLABLE Sphere object_hit(const Ray&, Point&, Vector3d<float>&, bool&) const;

#ifdef __CUDACC__
	__host__ void cuda_malloc_memcpy_pointer_members(Scene*);
	__host__ void cuda_memcpy_output();
	__host__ void cuda_free_pointer_members();
#endif

protected:
	Image* image_;
	Point* object_locations_;
	Sphere* objects_;
	Light* lights_;
	Point camera_location_;
	
	Point image_bl_location_; // Image's bottom left corner location
	float image_dims_[2];

	int n_objects_;
	int i_object_;


	int n_lights_;
	int i_light_;

#ifdef __CUDACC__
	Image* d_image_;
	Point* d_object_locations_;
	Sphere* d_objects_;
	Light* d_lights_;
#endif


};

#endif
