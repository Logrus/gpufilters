#include "lkopgpu.h"
#include <helpers/CTensor.h>
//#include "flowToImage.h"
#include <nlm_texture/nlmf.h>
#ifndef __linux__
#include <iso646.h>
#include <helpers/ImageDisplay.h>
//#include <helpers/timer_c.h>
#endif
#include <iomanip>

#define PI 3.1415926535897932384626433832795028842
#define BLOCKSIZE 32

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef1;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef2;

__global__ void smoothKernel(float* d_smoothed, float* d_image, float* d_filter, int x_size, int y_size, int filter_radius)
{
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;
	
	float result(0);

	for (int py = -filter_radius; py <= filter_radius; ++py)
		for (int px = -filter_radius; px <= filter_radius; ++px) {

			// Implement Neumann Boundaies
			int image_x = CLIP(0, xId + px, x_size - 1);
			int image_y = CLIP(0, yId + py, y_size - 1);

			float filterval = d_filter[filter_radius + px] * d_filter[filter_radius + py];
			float imageval = d_image[image_x + image_y*x_size];
			
			// Filter
			result = result + (filterval*imageval);
		}
	
	d_smoothed[yId*x_size + xId] = result;

}

__global__ void lukasKanadeKernel(float* u, float* v, float* IxIx, float* IyIy, float* IxIy, float* IxIt, float* IyIt, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;

	int idx = yId*x_size + xId;

	u[idx] = (-IxIt[idx] * IyIy[idx] + IxIy[idx] * IyIt[idx]) / (IxIx[idx] * IyIy[idx] - IxIy[idx] * IxIy[idx]);
	//__syncthreads();
	v[idx] = (-IyIt[idx] - IxIy[idx] * u[idx]) / IyIy[idx];

}

__global__ void computeDerivativeXKernel(float* d_out, float* d_image, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;

	int xm = CLIP(0, xId - 1, x_size - 1);
	int xp = CLIP(0, xId + 1, x_size - 1);

	d_out[yId*x_size + xId] = .5f*(d_image[yId*x_size + xp] - d_image[yId*x_size + xm]);

}

__global__ void computeDerivativeYKernel(float* d_out, float* d_image, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;

	int ym = CLIP(0, yId - 1, y_size - 1);
	int yp = CLIP(0, yId + 1, y_size - 1);

	d_out[yId*x_size + xId] = .5f*(d_image[yp*x_size + xId] - d_image[ym*x_size + xId]);

}

__global__ void differenceKernel(float* d_second, float* d_first, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;
	//float tmp = d_second[yId*x_size + xId];
	d_second[yId*x_size + xId] = d_second[yId*x_size + xId] - d_first[yId*x_size + xId];

}

__global__ void multiplyKernel(float* d_out, float* mat1, float* mat2, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	if (xId >= x_size || yId >= y_size)
		return;

	d_out[yId*x_size + xId] = mat1[yId*x_size + xId] * mat2[yId*x_size + xId];
}

__global__ void nlmKernelNaiveLK(
	float *d_image,
	float *d_out,
	float *filter,
	int x_size,
	int y_size,
	int window_radius,
	int patch_radius,
	float inv_sqr_sigma){


	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	int center = patch_radius;
	float new_value(0); float normalizer(0);

	// Cut boundary of the window if it exceeds image size
	int wxb = fmaxf(0, xId - window_radius); // window x begin
	int wyb = fmaxf(0, yId - window_radius); // window y begin
	int wxe = fminf(x_size - 1, xId + window_radius); // window x end
	int wye = fminf(y_size - 1, yId + window_radius); // window y end

	// go through window
	for (int wx = wxb; wx <= wxe; wx++){
		for (int wy = wyb; wy <= wye; wy++){

			float distance(0);
			// go through each patch
			for (int px = -patch_radius; px <= patch_radius; px++)
				for (int py = -patch_radius; py <= patch_radius; py++){

					// Main patch
					int px1 = CLIP(0, xId + px, x_size - 1);//fminf(x_size - 1, fmaxf(0, xId + px));
					int py1 = CLIP(0, yId + py, y_size - 1);//fminf(y_size - 1, fmaxf(0, yId + py));

					// Patch in the window
					int px2 = CLIP(0, wx + px, x_size - 1);//fminf(x_size - 1, fmaxf(0, wx + px));
					int py2 = CLIP(0, wy + py, y_size - 1);//fminf(y_size - 1, fmaxf(0, wy + py));

					float tmp = d_image[px1 + py1*x_size] - d_image[px2 + py2*x_size];
					distance += tmp*tmp*filter[center + px] * filter[center + py];


				}// go through each patch

			float w = exp(-distance * inv_sqr_sigma);
			new_value += w*d_image[wx + wy*x_size];
			normalizer += w;
		}
	}

	d_out[yId*x_size + xId] = new_value / normalizer;
}

__global__ void nlmKernelTemporal(
	float *d_result,
	float *u,
	float *v,
	float *filter,
	int x_size,
	int y_size,
	int window_radius,
	int patch_radius,
	float inv_sqr_sigma){


	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	int idx = yId*x_size + xId;
	int center = patch_radius;
	float new_value(0); float normalizer(0);

	// Cut boundary of the window if it exceeds image size
	int wxb = fmaxf(0, xId - window_radius); // window x begin
	int wyb = fmaxf(0, yId - window_radius); // window y begin
	int wxe = fminf(x_size - 1, xId + window_radius); // window x end
	int wye = fminf(y_size - 1, yId + window_radius); // window y end

	// go through window
	for (int wy = wyb; wy <= wye; wy++){
		for (int wx = wxb; wx <= wxe; wx++)
			for (int t = 0; t < 2; t++)
			{

				float distance(0);
				// go through each patch
				for (int py = -patch_radius; py <= patch_radius; py++)
					for (int px = -patch_radius; px <= patch_radius; px++){

						// Coordinates for Optical Flow
						int px1 = CLIP(0, xId + px, x_size - 1);
						int py1 = CLIP(0, yId + py, y_size - 1);
						int px2 = CLIP(0, wx + px, x_size - 1);
						int py2 = CLIP(0, wy + py, y_size - 1);

						// Coordinates for textures
						float px1n = px1 / (float)x_size;
						float py1n = py1 / (float)y_size;
						float px2n = px2 / (float)x_size;
						float py2n = py2 / (float)y_size;

						float un = -u[px2 + py2*x_size] / (float)x_size;
						float vn = -v[px2 + py2*x_size] / (float)y_size;

						float t1_v1 = tex2D(texRef2, px1n, py1n);
						float t1_v2 = tex2D(texRef2, px2n, py2n);
						float t2_v1 = tex2D(texRef1, px2n + un, py2n + vn);

						float tmp = t1_v1 - t1_v2*(1 - t) - t2_v1*t; 
						distance += tmp*tmp*filter[center + px] * filter[center + py];

					}// go through each patch
				
				float w = exp(-distance * inv_sqr_sigma);
				float tx = wx / (float)x_size; 
				float ty = wy / (float)y_size;
				new_value += w*(t*tex2D(texRef1, tx, ty)+(1-t)*tex2D(texRef2, tx, ty));
				normalizer += w;

			}
	}// go through window

	d_result[idx] = new_value / normalizer;
}

__global__ void nlmKernelTemporalWindowed(
	float *d_result,
	float *u,
	float *v,
	float *filter,
	int x_size,
	int y_size,
	int window_radius,
	int patch_radius,
	float inv_sqr_sigma){


	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	int idx = yId*x_size + xId;
	int center = patch_radius;
	float new_value(0); float normalizer(0);

	// go through time
	for (int t = 0; t < 2; t++){

		int x_u = xId;
		int y_v = yId;

		// Cut boundary of the window if it exceeds image size
		int wxb = fmaxf(0, x_u - window_radius); // window x begin
		int wyb = fmaxf(0, y_v - window_radius); // window y begin
		int wxe = fminf(x_size - 1, x_u + window_radius); // window x end
		int wye = fminf(y_size - 1, y_v + window_radius); // window y end

		for (int wy = wyb; wy <= wye; wy++){
			for (int wx = wxb; wx <= wxe; wx++)
			{
				float distance(0);

				// go through each patch
				for (int py = -patch_radius; py <= patch_radius; py++)
					for (int px = -patch_radius; px <= patch_radius; px++){

						// Coordinates for Optical Flow
						int px1 = CLIP(0, xId + px, x_size - 1);
						int py1 = CLIP(0, yId + py, y_size - 1);
						int px2 = CLIP(0, wx + px, x_size - 1);
						int py2 = CLIP(0, wy + py, y_size - 1);

						// Coordinates for textures
						float px1n = px1 / (float)x_size;
						float py1n = py1 / (float)y_size;
						float px2n = (px2+u[idx]*t) / (float)x_size;
						float py2n = (py2+v[idx]*t) / (float)y_size;

						float t1_v1 = tex2D(texRef2, px1n, py1n);
						
						float t1_v2(0);
						if (!t)
							t1_v2 = tex2D(texRef2, px2n, py2n);
						else
							t1_v2 = tex2D(texRef1, px2n, py2n);

						float tmp = t1_v1 - t1_v2; 
						distance += tmp*tmp*filter[center + px] * filter[center + py];


					}// go through each patch

				float w = exp(-distance * inv_sqr_sigma);
				float tx = wx / (float)x_size;
				float ty = wy / (float)y_size;
				new_value += w*(t * tex2D(texRef1, tx, ty) + (1 - t)*tex2D(texRef2, tx, ty));
				normalizer += w;

			}

		}

	}


	d_result[idx] = new_value / normalizer;
}


NLMLK::NLMLK(CVector < CMatrix<float> > sequence_in, int patch_radius, int sqr_sigma)
	: patch_radius(patch_radius), sqr_sigma(sqr_sigma)
{
	sequence = sequence_in;
	this->inv_sqr_sigma = 1.f / sqr_sigma;
	if (sequence.size() < 2)
		{std::cerr << "Sequence must contain more than two images"; exit(1); }

	this->x_size = sequence[0].xSize();
	this->y_size = sequence[0].ySize();
	this->image_size = x_size*y_size;

	checkCudaErrors(cudaMalloc(&d_firstImage, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&d_secondImage, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&d_firstUntouched, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&d_secondUntouched, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&It, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&Ix, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&Iy, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&u, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&v, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&IxIt, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&IyIt, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&IxIx, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&IyIy, sizeof(float)*image_size));
	checkCudaErrors(cudaMalloc(&IxIy, sizeof(float)*image_size));
	
	// Default optical flow should be 0
	cudaMemset(u, 0, sizeof(float)*image_size);
	cudaMemset(v, 0, sizeof(float)*image_size);

	// Default values for LukasKanade OF
	this->smoothingSigma = 5.0;
	this->precision = 3;
	this->filter_width = 2*std::ceil(precision*smoothingSigma)+1;
	this->filter_radius = (int) filter_width / 2;

	// Parameters for kernel initializations
	this->block = dim3(BLOCKSIZE, BLOCKSIZE, 1);
	this->grid.x = DIV_UP(x_size, (float)block.x);
	this->grid.y = DIV_UP(y_size, (float)block.y);

};

#if not defined(__linux__) || defined(_DEBUG)
ImageDisplay::ImageDisplay Display;
#endif


__inline__ void swap_pointers(float* &a, float* &b){ float * c = a; a = b; b = c; }

// We need them inside, could be cleaned up later
namespace lkof{
	// computes the mean squared error
	float mse(const CMatrix<float>& imgA, const CMatrix<float>& imgB)
	{
		float sqr_err = 0;
		for (int i = 0; i < imgA.size(); ++i)
		{
			float diff = imgA.data()[i] - imgB.data()[i];
			sqr_err += diff*diff;
		}
		return sqr_err / imgA.size();
	}


	// computes the peak signal to noise ratio for 8 bit images
	float psnr_8bit(const CMatrix<float>& noise_free, const CMatrix<float>& noisy)
	{
		float max_signal = 255;
		float psnr = 10.f*std::log10(max_signal*max_signal / mse(noise_free, noisy));
		return psnr;
	}
}


void NLMLK::computeOF()
{
#ifndef __linux__
	//timer::start("lkof");
#endif
	CMatrix<float> result(x_size, y_size);
	/*cudaStream_t s1, s2, s3, s4;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);
	cudaStreamCreate(&s3);
	cudaStreamCreate(&s4);*/
	float psnr_sum(0);
	// Compute filters
	float* h_filter = new float[2 * patch_radius + 1];
	float* center = h_filter + patch_radius;
	for (int x = -patch_radius; x <= patch_radius; ++x)
		*(center + x) = std::exp(-0.5*x*x / (patch_radius*patch_radius));


	float* h_lk_filter = new float[filter_width];
	center = h_lk_filter + filter_radius;
	for (int x = -filter_radius; x <= filter_radius; ++x)
		*(center + x) = std::exp(-0.5* x*x / (smoothingSigma*smoothingSigma)) / (smoothingSigma*sqrtf(2.f*PI));


	// Filter for Lukas Kanade Optical Flow
	float* d_lk_filter;
	checkCudaErrors(cudaMalloc(&d_lk_filter, sizeof(float)*filter_width));
	checkCudaErrors(cudaMemcpy(d_lk_filter, h_lk_filter, sizeof(float)*filter_width, cudaMemcpyHostToDevice));

	float * d_out;
	cudaMalloc(&d_out, sizeof(float) * image_size);

	// Filter for denoising
	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*filter_width));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filter_width, cudaMemcpyHostToDevice));


	// Denoise the first image with naive kernel
	checkCudaErrors(cudaMemcpy(d_firstImage, sequence[0].data(), sizeof(float)*image_size, cudaMemcpyHostToDevice));
	nlmKernelNaiveLK << <grid, block >> >(d_firstImage, d_firstUntouched, d_filter, x_size, y_size, 7, 3, inv_sqr_sigma);
	checkCudaErrors(cudaMemcpy(result.data(), d_firstUntouched, sizeof(float)*image_size, cudaMemcpyDeviceToHost));
	psnr_sum += lkof::psnr_8bit(ground_truth[0], result);

	for (int k = 0; k < sequence.size()-1; ++k){

		// Copy pictures to the device
		//checkCudaErrors(cudaMemcpy(d_firstUntouched, sequence[k].data(), sizeof(float)*image_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_secondUntouched, sequence[k + 1].data(), sizeof(float)*image_size, cudaMemcpyHostToDevice));

		cudaBindTexture2D(0, texRef1, d_firstUntouched, x_size, y_size, sizeof(float)*x_size);
		cudaBindTexture2D(0, texRef2, d_secondUntouched, x_size, y_size, sizeof(float)*x_size);

		texRef1.addressMode[0] = cudaAddressModeMirror;
		texRef1.addressMode[1] = cudaAddressModeMirror;
		texRef1.filterMode = cudaFilterModeLinear;
		texRef1.normalized = true;
		texRef2.addressMode[0] = cudaAddressModeMirror;
		texRef2.addressMode[1] = cudaAddressModeMirror;
		texRef2.filterMode = cudaFilterModeLinear;
		texRef2.normalized = true;

#define LK_ON
#ifdef LK_ON
		// ======== LUKAS KANADE OF ============================
		// Compute LK from previous denoised frame or from noisy image
		//checkCudaErrors(cudaMemcpy(d_firstImage, sequence[k].data(), sizeof(float)*image_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_firstImage, d_firstUntouched, sizeof(float)*image_size, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(It, sequence[k + 1].data(), sizeof(float)*image_size, cudaMemcpyHostToDevice));

		smoothKernel << <grid, block >> >(d_out, d_firstImage, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, d_firstImage);
		

		smoothKernel << <grid, block >> >(d_out, It, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, It);
		

		computeDerivativeXKernel << <grid, block >> >(Ix, d_firstImage, x_size, y_size);
		computeDerivativeYKernel << <grid, block >> >(Iy, d_firstImage, x_size, y_size);
		differenceKernel << <grid, block >> >(It, d_firstImage, x_size, y_size);
		
		multiplyKernel << <grid, block >> >(IxIx, Ix, Ix, x_size, y_size);
		multiplyKernel << <grid, block >> >(IyIy, Iy, Iy, x_size, y_size);
		multiplyKernel << <grid, block >> >(IxIy, Ix, Iy, x_size, y_size);
		multiplyKernel << <grid, block >> >(IxIt, Ix, It, x_size, y_size);
		multiplyKernel << <grid, block >> >(IyIt, Iy, It, x_size, y_size);
		
		smoothKernel << <grid, block >> >(d_out, IxIx, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, IxIx);
	
		smoothKernel << <grid, block >> >(d_out, IyIy, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, IyIy);
		
		smoothKernel << <grid, block >> >(d_out, IxIy, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, IxIy);
		
		smoothKernel << <grid, block >> >(d_out, IxIt, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, IxIt);
		
		smoothKernel << <grid, block >> >(d_out, IyIt, d_lk_filter, x_size, y_size, filter_radius);
		swap_pointers(d_out, IyIt);
		
		lukasKanadeKernel << <grid, block >> >(u, v, IxIx, IyIy, IxIy, IxIt, IyIt, x_size, y_size);

		// ======== / LUKAS KANADE OF / ============================
#endif
		nlmKernelTemporalWindowed << < grid, block >> > (d_out, u, v, d_filter, x_size, y_size, 10, 3, inv_sqr_sigma);
		swap_pointers(d_out, d_firstUntouched); // Use denoised result as new texture1
		checkCudaErrors(cudaMemcpy(result.data(), d_firstUntouched, sizeof(float)*image_size, cudaMemcpyDeviceToHost));

#ifndef __linux__
		CTensor<float> disp(x_size, y_size, 2);
		float * h_u, *h_v;
		h_u = new float[image_size];
		h_v = new float[image_size];
		checkCudaErrors(cudaMemcpy(h_u, u, sizeof(float)*image_size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_v, v, sizeof(float)*image_size, cudaMemcpyDeviceToHost));
		for (int j = 0; j < y_size; ++j)
			for (int i = 0; i < x_size; ++i)
			{
				disp(i, j, 0) = h_u[j*x_size + i];
				disp(i, j, 1) = h_v[j*x_size + i];
			}
		CTensor<float> dispRGB(x_size, y_size, 3);
		//flowToImage(disp, dispRGB);
		Display.Display(dispRGB, "OF");
		dispRGB.writeToPPM(SSTR("resources/out/of" << k << ".ppm").c_str());
		delete[] h_u, h_v;
		Display.Display(result, "Res");
		result.writeToPGM(SSTR("resources/out/sm" << k << ".pgm").c_str());
		Display.Display(sequence[k+1], "Noisy");
		Display.Display(ground_truth[k+1], "Ground");
#endif

		psnr_sum += lkof::psnr_8bit(ground_truth[k + 1], result);
		std::cout << "PSNR denoised: " << psnr_sum/(k+1) << " dB\n";
	}
#ifndef __linux__
	//timer::stop("lkof");
	//float time = timer::elapsed();
	//std::cout << "Time Averaged: " << time / sequence.size() << " dB\n";
#endif
	std::cout << "PSNR Averaged: " << psnr_sum / sequence.size() << " dB\n";

	delete[] h_filter;
	cudaFree(d_lk_filter);
}

NLMLK::~NLMLK(){

	checkCudaErrors(cudaFree(d_firstImage));
	checkCudaErrors(cudaFree(d_firstUntouched));
	checkCudaErrors(cudaFree(d_secondImage));
	checkCudaErrors(cudaFree(d_secondUntouched));
	checkCudaErrors(cudaFree(It));
	checkCudaErrors(cudaFree(Ix));
	checkCudaErrors(cudaFree(Iy));
	checkCudaErrors(cudaFree(u));
	checkCudaErrors(cudaFree(v));
	//float* IxIt, *IyIt, *IxIx, *IyIy, *IxIy;
	checkCudaErrors(cudaFree(IxIt));
	checkCudaErrors(cudaFree(IyIt));
	checkCudaErrors(cudaFree(IxIx));
	checkCudaErrors(cudaFree(IyIy));
	checkCudaErrors(cudaFree(IxIy));

	cudaDeviceReset();
};