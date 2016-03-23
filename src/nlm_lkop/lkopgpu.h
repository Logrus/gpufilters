#ifndef KOPGPU_H_
#define KOPGPU_H_
#include <cstdio>
#include <helpers/CMatrix.h>
#include <helpers/utils.h>
#include <helpers/auxcu.h>

class NLMLK
{
public:
	NLMLK(CVector < CMatrix<float> > sequence, int patch_radius, int sqr_sigma);
	~NLMLK();
	CVector< CMatrix<float> > sequence;
	CVector< CMatrix<float> > ground_truth;
	CMatrix<float> output;
	// Computes LukasKanade Optical Flow on GPU
	void computeOF();
	// Computes Non Local Means Filter on GPU
	void computeNLM();
	// Parameters for Non Local Means Filter
	int patch_radius;
	float sqr_sigma, inv_sqr_sigma;
	// Parameters of the pictures
	int x_size, y_size, image_size;
	float* d_firstImage, *d_secondImage;
	float* d_firstUntouched, *d_secondUntouched;
	float* d_filter;
	float* It, *Ix, *Iy, *u, *v;
	float* IxIt, *IyIt, *IxIx, *IyIy, *IxIy;
	float smoothingSigma;
	float precision;
	int filter_width, filter_radius;

	dim3 block;
	dim3 grid;
};

#endif