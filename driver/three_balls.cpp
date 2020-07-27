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


int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;

	// instantiate scene with 4 objects and 1 light
	const int n_obj = 4;
	const int n_lights = 1;
	Point cam_location = Point(0.f, 0.f, -1.f);
	Scene scene = Scene(cam_location, n_obj, n_lights);

	// -y  ^ z
	// ^  /
	// | /
	// |/
	// |----> x 

	// object 1 - ground
	Material obj_floor_mat = Material(Color(pow(.84f,2.f), pow(.8f,2.f), pow(.8f,2.f)), false, 1.f, 1.f, .2f, 0.2f);
	Sphere obj_floor = Sphere(1000.f, obj_floor_mat);
	Point obj_floor_loc(0.f, 1000.5f, 1.f);
	scene.add_object(obj_floor, obj_floor_loc);

	// object 2
	Material obj_mat = Material(Color(1.0f, 0.f, 0.f));
	Sphere obj = Sphere(.5f, obj_mat);
	Point obj_loc = Point(-2.f,0.f,1.f);
	scene.add_object(obj, obj_loc);

	// object 3
	obj_mat = Material(Color(pow(0.2f,2), 0.f, 1.f));
	obj = Sphere(.5f, obj_mat);
	obj_loc = Point(3.f,0.f,2.f);
	scene.add_object(obj, obj_loc);

	// object 4
	obj_mat = Material(Color(pow(0.2f, 2.f), pow(0.75f, 2.f), pow(0.2f, 2.f)));
	obj = Sphere(.5f, obj_mat);
	obj_loc = Point(-.5f,0.f,4.f);
	scene.add_object(obj, obj_loc);

	// light
	Color light_col = Color(pow(.7f,2.f), pow(.7f,2.f), pow(.8f,2.f));
	Point light1_loc = Point(10.f, -10.f, -2.f);
	Light light1 = Light(light1_loc, light_col);
	scene.add_light(light1);

	// render image
	printf("rendering... \n");
	Renderer engine = Renderer(&scene);
	engine.render();

	// write image
	printf("writing...\n");
#ifdef __CUDACC__
	scene.image()->write_ppm("three_balls_gpu.ppm");
#else
	scene.image()->write_ppm("three_balls_cpu.ppm");
#endif
	printf("Done! \n");

	return 0;
}