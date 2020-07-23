#include <stdio.h>
#include <math.h>

#include "scene.h"

Scene::Scene(Point camera_location, int n_objects, int n_lights) 
	:   camera_location_(0.f,0.f,0.f),
		image_bl_location_(0.f,0.f,0.f),
		n_objects_(n_objects),
		i_object_(0),
		n_lights_(n_lights),
		i_light_(0)
{
	camera_location_ = camera_location;

	float ar = 16.f/9.f; // aspect ration = width/height
	int height = 1000;
	int width = (int)(height*ar);
	image_ = new Image(height, width);

	// we align center of image with camera
	// image is at z=1 relative to camera
	image_dims_[0] = 2.f;
	image_dims_[1] = image_dims_[0]/ar;
	float image_z = +1.f;
	image_bl_location_ = camera_location_ + Point(-image_dims_[0]/2.f, -image_dims_[1]/2.f, image_z);

	object_locations_ = new Point[n_objects];
	objects_ = new Sphere[n_objects];
	lights_ = new Light[n_lights];
}

Scene::~Scene()
{
	delete image_;
	delete [] objects_;
	delete [] object_locations_;
	delete [] lights_;
}

void Scene::add_object(Sphere obj, Point obj_loc)
{
	objects_[i_object_] = obj;
	object_locations_[i_object_] = obj_loc;
	i_object_++;
}

void Scene::add_light(Light light)
{
	lights_[i_light_] = light;
	i_light_++;
}

CUDA_CALLABLE Image* Scene::image() const 
{
	return image_;
}

CUDA_CALLABLE Point Scene::pixel_location(int irow, int jcol) const
{
	// Image is a unit square
	// We need to get the pixel's relative location
	// from the bottom left corner and
	// scale it by the image's actual dimensions
	// before computng its absolute location
	Point p = image()->pixel_location(irow, jcol);
	p[0] *= image_dims_[0];
	p[1] *= image_dims_[1];
	//printf("%d %d: %f %f %f \n", irow, jcol, (image_bl_location_ + p).values()[0], (image_bl_location_ + p).values()[1], (image_bl_location_ + p).values()[2]);
	return image_bl_location_ + p;
}

CUDA_CALLABLE Point Scene::camera_location() const
{
	return camera_location_;
}

CUDA_CALLABLE int Scene::n_objects() const
{
	return n_objects_;
}

CUDA_CALLABLE int Scene::n_lights() const
{
	return n_lights_;
}

CUDA_CALLABLE Light Scene::light(int i) const
{
	return lights_[i];
}

CUDA_CALLABLE Sphere Scene::object_hit(const Ray& ray,  Point& hit_location, Vector3d<float>& normal_hit_dir, bool& is_hit) const
{
	Sphere closest_obj_hit;
	float min_dist = INFINITY;
	is_hit = false;

	for (int i = 0; i < n_objects_; i++) {
		Point location;
		float dist = objects_[i].hit_distance(object_locations_[i], ray, location);
		if (dist < min_dist) {
			is_hit = true;
			min_dist = dist;
			closest_obj_hit = objects_[i];
			hit_location = location;
			normal_hit_dir = location - object_locations_[i];
			normal_hit_dir.normalize();
		}
	}
	return closest_obj_hit;
}
