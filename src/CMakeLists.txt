include_directories(${project_include})

if (CUDA_ENABLED)
    set_source_files_properties(vector.cpp   PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(color.cpp    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(image.cpp    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(material.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(object.cpp   PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(ray.cpp      PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(scene.cpp    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ) 
    set_source_files_properties(renderer.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ) 
    set_source_files_properties(light.cpp    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(util.cpp     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

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
    util.cpp
)

if (CUDA_ENABLED)
    cuda_add_library(${lib_name} STATIC ${src})
else()
    add_library(${lib_name} STATIC ${src})
endif()
