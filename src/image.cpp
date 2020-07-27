#include <vector>
#include <cmath>
#include <stdio.h>

#include "image.h"
#include "util.h"

Image::Image(int height, int width) 
    : height_(height),
      width_(width)
{
    pixels_ = new Color[height_*width_];
    dx_ = 1.f/(width_ - 1.f);
    dy_ = 1.f/(height_ - 1.f);
}

Image::~Image()
{
	delete [] pixels_;
}

CUDA_CALLABLE void Image::set_pixel(int irow, int jcol, Color color)
{
	Color gamma_corr_color = Color(sqrtf(color[0]), sqrtf(color[1]), sqrtf(color[2]));
    pixels_[irow*width_ + jcol] = gamma_corr_color;
}

CUDA_CALLABLE Color Image::pixel(int irow, int jcol) const
{
    return pixels_[irow*width_ + jcol];
}

CUDA_CALLABLE void Image::dimensions(int& nrow, int &ncol) const
{
	nrow = height_;
	ncol = width_;
}

CUDA_CALLABLE Point Image::pixel_location(int irow, int jcol) const
{
	float x = (float) (jcol + random_unit_float())*dx_;
	float y = (float) (irow + random_unit_float())*dy_;
	return Point(x, y, 0.0f);
}

int Image::write_ppm(const char *fname) const
{
	FILE *fptr;
	fptr = fopen(fname, "w");

	fprintf(fptr, "P3\n");
	fprintf(fptr, "%d %d\n", width_, height_);
	fprintf(fptr, "255\n");
	for (int i = 0; i < height_; i++) {
		for (int j = 0; j < width_; j++)
		{
			Color color = pixel(i,j);
			#pragma unroll (3)
			for (int k = 0; k < 3; k++) {
				fprintf(fptr, "%d ", static_cast<int> (fmax(fmin(color[k], .999), 0)*256));
			}
		}
		// 1 line per row
		fprintf(fptr,"\n");
	}
	fclose(fptr);
	return 0; // success
}

#ifdef __CUDACC__
__host__ void Image::cuda_malloc_memcpy_pointer_members(Image* d_image) {

	CUDA_CALL(cudaMalloc((void**)&(d_pixels_), height_*width_*sizeof(Color)));
	// no need to copy pixels_ to d_pixels_ since the pixels are what we're going to compute on the GPU

	CUDA_CALL(cudaMemcpy(&(d_image->pixels_), &d_pixels_, sizeof(Color*), cudaMemcpyHostToDevice));
}

__host__ void Image::cuda_memcpy_output() {
	 CUDA_CALL(cudaMemcpy(pixels_, d_pixels_, height_*width_*sizeof(Color), cudaMemcpyDeviceToHost));
}

__host__ void Image::cuda_free_pointer_members() {
	 CUDA_CALL(cudaFree(d_pixels_));
}
#endif
