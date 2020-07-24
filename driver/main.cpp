#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector.h>

#include "scene.h"
#include "color.h"
#include "material.h"
#include "object.h"
#include "light.h"
#include "renderer.h"

#include "util.h"

std::vector<Color> nice_colors = 
{
	Color(176.f/255.f, 102.f/255.f,  96.f/255.f),
	Color(217.f/255.f, 168.f/255.f, 143.f/255.f),
	Color(234.f/255.f, 195.f/255.f, 184.f/255.f),
	Color(202.f/255.f, 143.f/255.f,  68.f/255.f),
	Color(219.f/255.f, 173.f/255.f, 114.f/255.f),
	Color(249.f/255.f, 211.f/255.f, 165.f/255.f),
	Color(171.f/255.f, 156.f/255.f, 115.f/255.f),
	Color(210.f/255.f, 190.f/255.f, 150.f/255.f),
	Color(227.f/255.f, 220.f/255.f, 192.f/255.f),
	Color( 94.f/255.f, 119.f/255.f,   3.f/255.f),
	Color(155.f/255.f, 175.f/255.f, 142.f/255.f)
};


Color random_color() {
	int irdn = rand() % nice_colors.size();
	return nice_colors[irdn];
}

int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;
	//srand (time(NULL));
	// instantiate scene
	Point cam_location = Point(0.f, 0.f, -1.f);
	// -y  ^ z
	// ^  /
	// | /
	// |/
	// |--> x 
	// create objects and add to scene
	// object 1

	// Create randomly placed objects and 1 light
	const int n_obj = 2;//30; 
	const int n_lights = 1;
	Scene scene = Scene(cam_location, n_obj+1, n_lights); // + 1 huge sphere as the base

	Material obj_floor_mat = Material(Color(.84f, .8f, .8f), 1.f, 1.f, .2f, 0.2f, false);
	Sphere obj_floor = Sphere(100.f, obj_floor_mat);
	Point obj_floor_loc(0.f, 100.5f, 1.f);
	scene.add_object(obj_floor, obj_floor_loc);
	//const float radius = .5f;
	//for (int i = 0; i < n_obj; i++) {
	//	Material obj_i_mat = Material(random_color(), 1.f, 1.f, 1.f, 0.5f, false);
	//	Sphere obj_i = Sphere(radius, obj_i_mat);
	//	float theta = 1.4f*random_float() -.69f;// -.69f + 1.4f*(float)i/(float) n_z;
	//	float z = 35.f*random_float() + 4.f;// 5.f + 30.f* (float)k/(float) n_z + 3.f*random_float();
	//	Point obj_i_loc = Point(tan(theta)*z,0.f, z);
	//	scene.add_object(obj_i, obj_i_loc);
	//}
	Material obj_mat = Material(Color(1.0f, 0.f, 0.f));
	Sphere obj = Sphere(.5f, obj_mat);
	Point obj_loc = Point(-2.f,0.f,1.f);
	scene.add_object(obj, obj_loc);

	obj_mat = Material(Color(0.f, 0.f, 1.f));
	obj = Sphere(.5f, obj_mat);
	obj_loc = Point(2.f,0.f,2.f);
	scene.add_object(obj, obj_loc);

	// TODO: create a bunch of random objects (reflective and not)

	// light 1
	Color light_col = Color(.8f, .8f, .8f);
	Point light1_loc = Point(10.f, -10.f, -2.f);
	Light light1 = Light(light1_loc, light_col);
	scene.add_light(light1);
	// light 2
	Point light2_loc = Point(0.f, -10.f, 1.f);
	Light light2 = Light(light2_loc, light_col);
	//scene.add_light(light2);

	// render image
	printf("rendering... \n");
	Renderer engine = Renderer(&scene);
	engine.render();

	// write image
	printf("writing...\n");
#ifdef __CUDACC__
	scene.image()->write_ppm("demo_gpu.ppm");
#else
	scene.image()->write_ppm("demo_cpu.ppm");
#endif
	printf("Done! \n");

	return 0;
}