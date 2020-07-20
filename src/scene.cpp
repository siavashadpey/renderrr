#include <math.h>

#include "scene.h"

Scene::Scene(Point camera_location) 
	:	camera_location_(camera_location), 
		image_bl_location_(0.,0.,0.)
{
	double ar = 16./9.; // aspect ration = width/height
	int height = 1000;
	int width = (int)(height*ar);
	image_ = new Image(height, width);

	// we align center of image with camera
	// image is at z=1 relative to camera
	image_dims_[0] = 2.;
	image_dims_[1] = image_dims_[0]/ar;
	double image_z = +1;
	image_bl_location_ = camera_location_ + Point(-image_dims_[0]/2., -image_dims_[1]/2., image_z);
}

Scene::~Scene()
{
	delete image_;
}

void Scene::add_object(Sphere *obj, Point* obj_loc)
{
	objects_.push_back(obj);
	object_locations_.push_back(obj_loc);
}

void Scene::add_light(Light *light)
{
	lights_.push_back(light);
}

Image* Scene::image() const 
{
	return image_;
}

Point Scene::pixel_location(int irow, int jcol) const
{
	// Image is a unit square
	// We need to get the pixel's relative location
	// from the bottom left corner and
	// scale it by the image's actual dimensions
	// before computng its absolute location
	Point p = image()->pixel_location(irow, jcol);
	double* vals = p.values();
	vals[0] *= image_dims_[0];
	vals[1] *= image_dims_[1];
	//printf("%d %d: %f %f %f \n", irow, jcol, (image_bl_location_ + p).values()[0], (image_bl_location_ + p).values()[1], (image_bl_location_ + p).values()[2]);
	return image_bl_location_ + p;
}

Point Scene::camera_location() const
{
	return camera_location_;
}

int Scene::n_objects() const
{
	return objects_.size();
}

int Scene::n_lights() const
{
	return lights_.size();
}

Light* Scene::light(int i) const
{
	return lights_[i];
}

Sphere* Scene::object_hit(const Ray& ray,  Point& hit_location, Vector3d<double>& normal_hit_dir) const
{
	Sphere* closest_obj_hit = nullptr;
	double min_dist = INFINITY;

	for (unsigned int i = 0; i < objects_.size(); i++) {
		Point location;
		double dist = objects_[i]->hit_distance(*object_locations_[i], ray, location);
		if (dist < min_dist) {
			min_dist = dist;
			closest_obj_hit = objects_[i];
			hit_location = location;
			normal_hit_dir = location - *object_locations_[i];
			normal_hit_dir.normalize();
		}
	}
	return closest_obj_hit;
}
