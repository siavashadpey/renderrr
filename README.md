# renderrr
A GPU implementation of a simple ray tracer. Some of the ray tracing features are from [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#rays,asimplecamera,andbackground/sendingraysintothescene) and some are from [Building a Ray Tracer in Python](https://www.youtube.com/watch?v=KaCe63v4D_Q).


## Running the code

First clone the code:

```console
git clone https://github.com/siavashadpey/renderrr.git
```

To compile the NVIDIA GPU implementation of the code, run the following commands from the main directory `renderrr`:

```console
mkdir build_parallel
cd build_parallel
cmake -DCUDA=ON ..
make
```

To compile the CPU implementation of the code, run the following commands from the main directory `renderrr`:

```console
mkdir build_serial
cd build_serial
cmake ..
make
```

Finally, run the code from `build_parallel` `build_serial` to generate the ppm file:
```console
./driver/three_balls
```
