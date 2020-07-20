#include <stdio.h>

#include "scene.h"
#include "color.h"
#include "material.h"
#include "object.h"
#include "light.h"
#include "renderer.h"

int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;

	// instantiate scene
	Point cam_location = Point(0., -0.35, -1.);
	Scene scene = Scene(cam_location);
	// -y  ^ z
	// ^  /
	// | /
	// |/
	// |--> x 
	// create objects and add to scene
	// object 1
	Color obj1_col = Color(1., 0., 0.);
	Material obj1_mat = Material(obj1_col);
	Sphere obj1 = Sphere(0.6, &obj1_mat);
	Point obj1_loc(2., -0.1, 3.);
	scene.add_object(&obj1, &obj1_loc);
	// object 2
	Color obj2_col = Color(0., 1., 0.);
	Material obj2_mat = Material(obj2_col);
	Sphere obj2 = Sphere(0.6, &obj2_mat);
	Point obj2_loc(-3., -0.1, 5.);
	scene.add_object(&obj2, &obj2_loc);
	// object 3 - huge object
	Color obj3_col = Color(.25, .25, .25);
	Material obj3_mat = Material(obj3_col, 1., 1., 0.2, 0.2);
	Sphere obj3 = Sphere(10000., &obj3_mat);
	Point obj3_loc(0., 10000.5, 1.);
	scene.add_object(&obj3, &obj3_loc);

	Color obj4_col = Color(0., 0., 1.);
	Material obj4_mat = Material(obj4_col);
	Sphere obj4 = Sphere(.5, &obj4_mat);
	Point obj4_loc(0., -0.1, 4.);
	//scene.add_object(&obj4, &obj4_loc);

	// create and add lights
	// light 1
	Color white = Color(1., 1., 1.);
	Point light1_loc = Point(-3, -2.5, 5.);
	Light light1 = Light(light1_loc, white);
	scene.add_light(&light1);
	// light 2
	Point light2_loc = Point(-2, -2, -1.);
	Light light2 = Light(light2_loc, white);
	//scene.add_light(&light2);

	// render image
	Renderer engine = Renderer(&scene);
	engine.render();

	// write image
	scene.image()->write_ppm("demo.ppm");
	
	printf("Done! \n");

	return 0;
}

/**
TODO:
	1) Parallelize on CUDA
**/