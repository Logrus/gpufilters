#include "nlmf.h"
#include <cstdio>
#include <helpers/CMatrix.h>
#include <helpers/utils.h>
#include <helpers/auxcu.h>


texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void nlmKernelNaiveTexture(
	float *d_out,
	float *filter,
	int x_size,
	int y_size,
	int window_radius,
	int patch_radius,
	float inv_sqr_sigma){

	// Coordinates in global memory
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	extern __shared__ float sfilter[];
	for (int i = 0; i <= 2 * patch_radius; i++)
		sfilter[i] = filter[i];

	int center = patch_radius;

	float new_value(0); float normalizer(0);

	// Cut boundary of the window if it exceeds image size
	float wxb = xId - window_radius;// window x begin
	float wyb = yId - window_radius;// window y begin
	float wxe = xId + window_radius;// window x end
	float wye = yId + window_radius;// window y end

	// go through window
	for (float wy = wyb; wy <= wye; wy++){
		for (float wx = wxb; wx <= wxe; wx++){

			float distance(0);
			// go through each patch
			for (float py = -patch_radius; py <= patch_radius; py++)
				for (float px = -patch_radius; px <= patch_radius; px++){

					// Main patch
					float px1 = xId + px;
					float py1 = yId + py;
					// Patch in the window
					float px2 = wx + px;
					float py2 = wy + py;

					float tmp = tex2D(texRef, px1 + 0.5f, py1 + 0.5f) - tex2D(texRef, px2 + 0.5f, py2 + 0.5f);
					distance += tmp*tmp*sfilter[center + int(px)] * sfilter[center + int(py)];


				}// go through each patch

			float w = __expf(-distance * inv_sqr_sigma);
			new_value += w*tex2D(texRef, wx + 0.5f, wy + 0.5f);
			normalizer += w;
		}
	}// go through window

	d_out[yId*x_size + xId] = new_value / normalizer;
}

__global__ void nlmKernelNaive(float *d_image,
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
	for (int wy = wyb; wy <= wye; wy++){
		for (int wx = wxb; wx <= wxe; wx++){

			float distance(0);
			// go through each patch
			for (int py = -patch_radius; py <= patch_radius; py++)
				for (int px = -patch_radius; px <= patch_radius; px++){

					// Main patch
					int px1 = CLIP(0, xId + px, x_size - 1);
					int py1 = CLIP(0, yId + py, y_size - 1);

					// Patch in the window
					int px2 = CLIP(0, wx + px, x_size - 1);
					int py2 = CLIP(0, wy + py, y_size - 1);

					float tmp = d_image[px1 + py1*x_size] - d_image[px2 + py2*x_size];
					distance += tmp*tmp*filter[center + px] * filter[center + py];


				}// go through each patch

			float w = exp(-distance * inv_sqr_sigma);
			new_value += w*d_image[wx + wy*x_size];
			normalizer += w;
		}
	}// go through window

	d_out[yId*x_size + xId] = new_value / normalizer;
}


CMatrix<float> nonLocalMeanCudaNaive(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma){

	CMatrix<float> result(image.xSize(), image.ySize(), 0);

	float inv_sqr_sigma = 1. / sqr_sigma;

	int image_size = image.size();//image.xSize() * image.ySize();

	// create a gauss lut for 1D
	float* h_filter = new float[2 * patch_radius + 1];
	float* center = h_filter + patch_radius;
	for (int x = -patch_radius; x <= patch_radius; ++x)
		*(center + x) = std::exp(-0.5*x*x / (patch_radius*patch_radius));


	// Allocate memory on the device
	float *d_image, *d_filter, *d_out;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float) * image_size));
	checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float) * (2 * patch_radius + 1)));

	// Copy to the device
	checkCudaErrors(cudaMemcpy(d_image, image.data(), sizeof(float) * image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * (2 * patch_radius + 1), cudaMemcpyHostToDevice));

	// Compute kernel
	dim3 block(32, 32, 1);
	dim3 grid;
	grid.x = DIV_UP(image.xSize(), (float)block.x);
	grid.y = DIV_UP(image.ySize(), (float)block.y);
	nlmKernelNaive << < grid, block >> >(d_image, d_out, d_filter, image.xSize(), image.ySize(), window_radius, patch_radius, inv_sqr_sigma);

	// Copy result from the device
	checkCudaErrors(cudaMemcpy((void*)result.data(), (void*)d_out, sizeof(float) * result.size(), cudaMemcpyDeviceToHost));

	//cudaDeviceSynchronize(); 
	//checkCudaErrors(cudaGetLastError());
	cudaFree(d_image);
	cudaFree(d_out);
	cudaFree(d_filter);
	checkCudaErrors(cudaDeviceReset());
	return result;
}


CMatrix<float> nonLocalMeanCudaNaiveTexture(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma){

	CMatrix<float> result(image.xSize(), image.ySize(), 0);

	float inv_sqr_sigma = 1. / sqr_sigma;

	int image_size = image.size();//image.xSize() * image.ySize();

	// create a gauss lut for 1D
	float* h_filter = new float[2 * patch_radius + 1];
	float* center = h_filter + patch_radius;
	for (int x = -patch_radius; x <= patch_radius; ++x)
		*(center + x) = std::exp(-0.5*x*x / (patch_radius*patch_radius));


	// Allocate memory on the device
	float *d_image, *d_filter, *d_out;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float) * image_size));
	checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float) * (2 * patch_radius + 1)));

	// Copy to the device
	checkCudaErrors(cudaMemcpy(d_image, image.data(), sizeof(float) * image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * (2 * patch_radius + 1), cudaMemcpyHostToDevice));

	// Bind the memory to the texture using cudaBindTexture()
	cudaBindTexture2D(0, texRef, d_image, image.xSize(), image.ySize(), image.xSize()* sizeof(float));

	// Setup the texture parameters 
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModePoint;
	texRef.normalized = false;

	// Compute kernel
	dim3 block(32, 32, 1);
	dim3 grid;
	grid.x = DIV_UP(image.xSize(), (float)block.x);
	grid.y = DIV_UP(image.ySize(), (float)block.y);
	int sfiltsize = (2 * patch_radius + 1) * sizeof(float);
	nlmKernelNaiveTexture << < grid, block, sfiltsize >> >(d_out, d_filter, image.xSize(), image.ySize(), window_radius, patch_radius, inv_sqr_sigma);

	// Copy result from the device
	checkCudaErrors(cudaMemcpy((void*)result.data(), (void*)d_out, sizeof(float) * result.size(), cudaMemcpyDeviceToHost));

	//cudaDeviceSynchronize(); 
	//checkCudaErrors(cudaGetLastError());
	cudaFree(d_image);
	cudaFree(d_out);
	cudaFree(d_filter);
	cudaUnbindTexture(texRef);
	checkCudaErrors(cudaDeviceReset());
	return result;
}