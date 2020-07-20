#include <vector>
#include <cmath>
#include <stdio.h>

#include "image.h"

Image::Image(int height, int width) 
    : height_(height),
      width_(width)
{
    pixels_ = new Color*[height_];
    for (int i = 0; i < height_; i++) {
    	pixels_[i] = new Color[width_];
    }
    dx_ = 1.f/(width_ - 1.f);
    dy_ = 1.f/(height_ - 1.f);
}

Image::~Image()
{
	for (int i = 0; i < height_; i++) {
		delete [] pixels_[i];
	}
	delete [] pixels_;
}

CUDA_CALLABLE void Image::set_pixel(int irow, int jcol, Color color)
{
    pixels_[irow][jcol] = color;
}

CUDA_CALLABLE Color* Image::pixel(int irow, int jcol) const
{
    return &pixels_[irow][jcol];
}

CUDA_CALLABLE void Image::dimensions(int& nrow, int &ncol) const
{
	nrow = height_;
	ncol = width_;
}

CUDA_CALLABLE Point Image::pixel_location(int irow, int jcol) const
{
	float x = (float) jcol*dx_;
	float y = (float) irow*dy_;
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
			float* rgb_ij = pixel(i,j)->RGB();
			#pragma unroll
			for (int k = 0; k < 3; k++) {
				fprintf(fptr, "%d ", (int)round(fmax(fmin(rgb_ij[k] * 255, 255), 0.f)));
			}
		}
		// 1 line per row
		fprintf(fptr,"\n");
	}
	fclose(fptr);
	return 0; // success
}