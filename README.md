# renderrr
A GPU implementation of a simple ray tracer.


## Compiling the code

First clone the code
```console
git clone 
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
