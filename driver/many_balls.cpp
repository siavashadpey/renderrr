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
	Color(pow( 239.f/255.f, 2.f), pow(  62.f/255.f, 2.f), pow( 91.f/255.f, 2.f)), 
	Color(pow( 246.f/255.f, 2.f), pow(  98.f/255.f, 2.f), pow( 121.f/255.f, 2.f)), 
	Color(pow( 111.f/255.f, 2.f), pow(  84.f/255.f, 2.f), pow( 149.f/255.f, 2.f)), 
	Color(pow(  63.f/255.f, 2.f), pow( 100.f/255.f, 2.f), pow( 126.f/255.f, 2.f)), 
	Color(pow( 104.f/255.f, 2.f), pow( 143.f/255.f, 2.f), pow( 173.f/255.f, 2.f)), 
	Color(pow(   0.f/255.f, 2.f), pow( 176.f/255.f, 2.f), pow( 178.f/255.f, 2.f)), 
	Color(pow( 140.f/255.f, 2.f), pow( 212.f/255.f, 2.f), pow( 122.f/255.f, 2.f)),
	Color(pow( 255.f/255.f, 2.f), pow( 163.f/255.f, 2.f), pow(  67.f/255.f, 2.f)),
	Color(pow( 234.f/255.f, 2.f), pow( 126.f/255.f, 2.f), pow(  93.f/255.f, 2.f)),
	Color(pow( 110.f/255.f, 2.f), pow(  81.f/255.f, 2.f), pow(  96.f/255.f, 2.f))
};


Color random_color() {
	int irdn = rand() % nice_colors.size();
	return nice_colors[irdn];
}

bool random_bool() {
	return rand() % 2;
}

int main(int argc, char const *argv[])
{
	(void) argc;
	(void) argv;
	
	srand (time(NULL));


	// -y  ^ z
	// ^  /
	// | /
	// |/
	// |--> x 

	// instantiate scene
	const int n_obj = 50; 
	const int n_lights = 1;
	Point cam_location = Point(0.f, -.8f, -1.f);
	Scene scene = Scene(cam_location, n_obj+1, n_lights); // + 1 huge sphere as the base
	// ground
	Material obj_floor_mat = Material(Color(pow(.84f,2.f), pow(.8f,2.f), pow(.8f,2.f)), false, 1.f, 1.f, .2f, 0.2f);
	Sphere obj_floor = Sphere(1000.f, obj_floor_mat);
	Point obj_floor_loc(0.f, 1000.f, 1.f);
	scene.add_object(obj_floor, obj_floor_loc);
	for (int i = -5; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			const float radius = 0.2f*random_unit_float() + 0.2f;
			Material obj_i_mat = Material(random_color(), random_bool());
			Sphere obj_i = Sphere(radius, obj_i_mat);
			Point obj_i_loc = Point((float)i + random_unit_float()/2.f, -radius, (float)j + random_unit_float()/2.f);
			scene.add_object(obj_i, obj_i_loc);
		}
	}

	// light 1
	Color light_col = Color(pow(.7f,2.f), pow(.7f,2.f), pow(.8f,2.f));
	Point light1_loc = Point(-10.f, -10.f, -2.f);
	Light light1 = Light(light1_loc, light_col);
	scene.add_light(light1);

	// render image
	printf("rendering... \n");
	Renderer engine = Renderer(&scene);
	engine.render();

	// write image
	printf("writing...\n");
#ifdef __CUDACC__
	scene.image()->write_ppm("many_balls_gpu.ppm");
#else
	scene.image()->write_ppm("many_balls_cpu.ppm");
#endif
	printf("Done! \n");

	return 0;
}