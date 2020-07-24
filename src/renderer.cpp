#include <stdio.h>
#include <math.h>

#include "renderer.h"

#ifdef __CUDACC__
__global__ void set_pixel_color_kernel(Renderer* renderer, int nrow, int ncol) {
	int i = blockDim.y*blockIdx.y + threadIdx.y;
	int j = blockDim.x*blockIdx.x + threadIdx.x;
	Point cl = renderer->scene()->camera_location();
	if ((i < nrow) and (j < ncol)) {
		Point pixel_loc = renderer->scene()->pixel_location(i, j);
		Vector3d<float> dir = cl.direction_to(pixel_loc);
		Ray ray = Ray(cl, dir);
		Color color;
		color = renderer->trace_ray(ray);
		renderer->set_pixel_color(i, j, color);
	}
}
#endif

Renderer::Renderer(Scene* scene)
{
	scene_ = scene;
}

Renderer::~Renderer()
{}

void Renderer::render()
{

	int nrow;
	int ncol;
	scene_->image()->dimensions(nrow, ncol);

#ifdef __CUDACC__

	// allocate global memory and copy objects
	Renderer* d_renderer;
	CUDA_CALL(cudaMalloc((void**) &d_renderer, sizeof(Renderer) ));
	CUDA_CALL(cudaMemcpy(d_renderer, this, sizeof(Renderer), cudaMemcpyHostToDevice));

	// cudaMalloc & cudaMemcpy does not copy pointer data members of class objects
	// so we need to cudaMalloc & cudaMemcpy those explicitly
	// and repeat the process for pointer data members that
	// have pointer data members
	cuda_malloc_memcpy_pointer_members_(d_renderer);

	// setup grid
	dim3 block(NTHREADS, NTHREADS);
	dim3 grid( (ncol + NTHREADS - 1)/NTHREADS, (nrow + NTHREADS - 1)/NTHREADS );

	// launch kernel
	set_pixel_color_kernel<<<grid, block>>>(d_renderer, nrow, ncol);
	//CUDA_CALL( cudaPeekAtLastError());
	//CUDA_CALL( cudaDeviceSynchronize());

	// copy output
	cuda_memcpy_output_();

	// free device memory
	CUDA_CALL(cudaFree(d_renderer));
	cuda_free_pointer_members_();

#else

	Point cl = scene()->camera_location();
	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			Point pixel_loc = scene()->pixel_location(i, j);
			Vector3d<float> dir = cl.direction_to(pixel_loc);
			Ray ray = Ray(cl, dir);
			Color color;
			color = trace_ray(ray);
			set_pixel_color(i, j, color);
		}
	}
#endif
}

CUDA_CALLABLE Scene* Renderer::scene() const
{
	return scene_;
}

CUDA_CALLABLE Color Renderer::trace_ray(Ray &ray)
{
	//Point hit_location;
	//Vector3d<float> hit_normal_dir;
	//bool is_hit;
	//Sphere obj = scene()->object_hit(ray, hit_location, hit_normal_dir, is_hit);
//
	//Color color;
	//if (is_hit) {
	//	color_at_(obj, hit_location, hit_normal_dir, color);
	//	// trace ray reflected off of object
	//	if (iray < max_rays_) {
	//		Point new_position = hit_location + hit_normal_dir*0.0001f;
	//		Vector3d<float> new_dir = ray.direction() - hit_normal_dir*hit_normal_dir.dot(ray.direction())*2.f;
	//		new_dir.normalize();
	//		Ray reflected_ray = Ray(new_position, new_dir); 
	//		color += trace_ray(reflected_ray, iray+1)*obj.reflection_intensity();
	//	}
	//}
	//return color;

	Color tot_color;
	float reflection_intensity = 1.0f;
	for (int iray = 0; iray < max_rays_; iray++) {
		Point hit_location;
		Vector3d<float> hit_normal_dir;
		bool is_hit;
		Color color;
		Sphere obj = scene()->object_hit(ray, hit_location, hit_normal_dir, is_hit);
		if (is_hit) {
			color_at_(obj, hit_location, hit_normal_dir, color);
			tot_color += color*reflection_intensity;
			reflection_intensity *= obj.reflection_intensity();
			// trace ray reflected off of object
			Point new_position = hit_location + hit_normal_dir*0.0001f;
			Vector3d<float> new_dir = ray.direction() - hit_normal_dir*hit_normal_dir.dot(ray.direction())*2.f;
			new_dir.normalize();
			ray = Ray(new_position, new_dir);
			//color += trace_ray(reflected_ray, emitted_ray_counter+1)*obj.reflection_intensity();
		}
		else {
			return tot_color;
		}
	}
	return tot_color;
}


CUDA_CALLABLE void Renderer::color_at_(const Sphere& obj, const Point& hit_location, const Vector3d<float>& hit_normal_dir, Color& color) const 
{
	// ambient color
	color += obj.base_color()*obj.ambient_intensity();

	Point hit_to_cam = scene()->camera_location() - hit_location;
	for (int i = 0; i < scene()->n_lights(); i++) {
		Light current_light = scene()->light(i);

		// diffuse color - Lambert shading model
		Point hit_to_light = hit_location.direction_to(current_light.location());
		color += obj.base_color()*fmax(hit_to_light.dot(hit_normal_dir), 0.0f)*obj.diffuse_coefficient();
		
		// specular color - Phong shading model
		Point R = hit_to_light + hit_to_cam;
		R.normalize();
		color += current_light.color()*obj.specular_coefficient()*pow(fmax(hit_normal_dir.dot(R), 0.0f), 50.f);
	}
}

CUDA_CALLABLE void Renderer::set_pixel_color(int irow, int jcol, Color pixel_color)
{
	scene()->image()->set_pixel(irow, jcol, pixel_color);
}

CUDA_CALLABLE int Renderer::max_rays() const
{
	return max_rays_;
}

#ifdef __CUDACC__
__host__ void Renderer::cuda_malloc_memcpy_pointer_members_(Renderer* d_renderer) {
	CUDA_CALL(cudaMalloc((void**) &d_scene_, sizeof(Scene)));
	CUDA_CALL(cudaMemcpy(d_scene_, scene_, sizeof(Scene), cudaMemcpyHostToDevice ));

	scene_->cuda_malloc_memcpy_pointer_members(d_scene_);

	//  overwrite the addresses of the pointer's data members that were not explicitly handle
	CUDA_CALL(cudaMemcpy(&(d_renderer->scene_), &d_scene_, sizeof(Scene*), cudaMemcpyHostToDevice ));
}

__host__ void Renderer::cuda_memcpy_output_() {
	scene_->cuda_memcpy_output();
}

__host__ void Renderer::cuda_free_pointer_members_() {
	CUDA_CALL(cudaFree(d_scene_));
	scene_->cuda_free_pointer_members();
}
#endif
