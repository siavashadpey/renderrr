include_directories(${project_include})

set(src
	vector.cpp
	color.cpp
	image.cpp
	material.cpp
	object.cpp
	ray.cpp
	scene.cpp
	renderer.cpp
	light.cpp
)

add_library(${lib_name} ${src})
