#include <stdio.h>
#include <cassert>
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
	image_dims_[1] = 2.f;
	image_dims_[0] = image_dims_[1]*ar;
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
	assert(i_object_ < n_objects_);
	objects_[i_object_] = obj;
	object_locations_[i_object_] = obj_loc;
	i_object_++;
}

void Scene::add_light(Light light)
{
	assert(i_light_ < n_lights_);
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
		}
	}
	normal_hit_dir.normalize();
	return closest_obj_hit;
}

CUDA_CALLABLE bool Scene::is_intercepted(const Point& origin, const Point& dest) const {
	const float length = (origin - dest).magnitude();
	Ray ray = Ray(origin, origin.direction_to(dest));
	for (int i = 0; i < n_objects_; i++) {
		Point location;
		float dist = objects_[i].hit_distance(object_locations_[i], ray, location);
		if (dist < length) {
			return true;
		}
	}
	return false;
}

CUDA_CALLABLE Color Scene::sky_color(const Ray& ray) const {
	const float t = 0.5f*(-ray.direction()[1] + 1.0f);
	Color c;
	c = (Color(1.f,1.f,1.f)*(1.f-t) + Color(.5f,.7f,1.f)*t)*1.f;
	return c;
}

#ifdef __CUDACC__
__host__ void Scene::cuda_malloc_memcpy_pointer_members(Scene* d_scene) {

	CUDA_CALL(cudaMalloc((void**) &d_image_, sizeof(Image)));
	CUDA_CALL(cudaMemcpy(d_image_, image_, sizeof(Image), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void**) &d_object_locations_, n_objects_*sizeof(Point)));
	CUDA_CALL(cudaMemcpy(d_object_locations_, object_locations_, n_objects_*sizeof(Point), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void**) &d_objects_, n_objects_*sizeof(Sphere)));
	CUDA_CALL(cudaMemcpy(d_objects_, objects_, n_objects_*sizeof(Sphere), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc((void**) &d_lights_, n_lights_*sizeof(Light)));
	CUDA_CALL(cudaMemcpy(d_lights_, lights_, n_lights_*sizeof(Light), cudaMemcpyHostToDevice));

	image_->cuda_malloc_memcpy_pointer_members(d_image_);

	CUDA_CALL(cudaMemcpy(&(d_scene->image_), &d_image_, sizeof(Image*), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(&(d_scene->object_locations_), &d_object_locations_, sizeof(Point*), cudaMemcpyHostToDevice)); 
	CUDA_CALL(cudaMemcpy(&(d_scene->objects_), &d_objects_, sizeof(Sphere*), cudaMemcpyHostToDevice)); 
	CUDA_CALL(cudaMemcpy(&(d_scene->lights_), &d_lights_, sizeof(Light*), cudaMemcpyHostToDevice));
}

__host__ void Scene::cuda_memcpy_output() {
	image_->cuda_memcpy_output();
}

__host__ void Scene::cuda_free_pointer_members() {
	CUDA_CALL(cudaFree(d_image_));
	CUDA_CALL(cudaFree(d_object_locations_));
	CUDA_CALL(cudaFree(d_objects_));
	CUDA_CALL(cudaFree(d_lights_));
	image_->cuda_free_pointer_members();
}

#endif
