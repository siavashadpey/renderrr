#include <vector>
#include <cmath>
#include <stdio.h>

#include "image.h"

Image::Image(int height, int width) 
    : height_(height),
      width_(width)
{
    pixels_.resize(height);
    for (unsigned int i = 0; i < pixels_.size(); i++) {
        pixels_[i].resize(width);
    }
    dx_ = 1./(width - 1.);
    dy_ = 1./(height - 1.);
}

Image::~Image()
{}

void Image::set_pixel(int irow, int jcol, Color color)
{
    pixels_[irow][jcol] = color;
}

Color Image::pixel(int irow, int jcol) const
{
    return pixels_[irow][jcol];
}

void Image::dimensions(int& nrow, int &ncol) const
{
	nrow = pixels_.size();
	ncol = pixels_[0].size();
}

Point Image::pixel_location(int irow, int jcol) const
{
	double x = (double) jcol*dx_;
	double y = (double) irow*dy_;
	return Point(x, y, 0.0);
}


int Image::write_ppm(const char *fname) const
{
	FILE *fptr;
	fptr = fopen(fname, "w");

	fprintf(fptr, "P3\n");
	fprintf(fptr, "%d %d\n", width_, height_);
	fprintf(fptr, "255\n");
	for (unsigned int i = 0; i < pixels_.size(); i++) {
		for (unsigned int j = 0; j < pixels_[i].size(); j++)
		{
			double* rgb_ij = pixel(i,j).RGB();
			#pragma unroll
			for (unsigned int k = 0; k < 3; k++) {
				fprintf(fptr, "%d ", (int)round(fmax(fmin(rgb_ij[k] * 255, 255), 0)));
			}
		}
		// 1 line per row
		fprintf(fptr,"\n");
	}
	fclose(fptr);
	return 0; // success
}