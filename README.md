# renderrr
A GPU implementation of a simple ray tracer.


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
./driver/main
```
