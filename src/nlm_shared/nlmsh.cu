#include "nlmsh.h"
#include <cstdio>
#include <helpers/CMatrix.h>
#include <helpers/utils.h>
#include <helpers/auxcu.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void nlmKernelShared(
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

	int local_block_w = blockDim.x + 2 * patch_radius + 2 * window_radius;
	int local_block_h = blockDim.y + 2 * patch_radius + 2 * window_radius;
	int kernel_rx = patch_radius + window_radius;
	int kernel_ry = patch_radius + window_radius;

	// We are working with pixel xId,yId
	// We need to go through the window
	float new_value(0); float normalizer(0);

	//Allocate shared memory
	extern __shared__ float LocalBlock[];
	int SharedIndex = threadIdx.y * local_block_w;


	LocalBlock[SharedIndex + threadIdx.x] = tex2D(texRef, xId - kernel_rx + 0.5f, yId - kernel_ry + 0.5f);
	if (threadIdx.x + blockDim.x < local_block_w)
		LocalBlock[SharedIndex + threadIdx.x + blockDim.x] = tex2D(texRef, xId + (int)blockDim.x - kernel_rx + 0.5f, yId - kernel_ry + 0.5f);
	if (threadIdx.y < (kernel_ry * 2))
	{
		SharedIndex = (threadIdx.y + blockDim.y)  * local_block_w;
		LocalBlock[SharedIndex + threadIdx.x] = tex2D(texRef, xId - kernel_rx + 0.5f, yId + (int)blockDim.y - kernel_ry + 0.5f);
		if (threadIdx.x + blockDim.x < local_block_w)
			LocalBlock[SharedIndex + threadIdx.x + blockDim.x] = tex2D(texRef, xId + (int)blockDim.x - kernel_rx + 0.5f, yId + (int)blockDim.y - kernel_ry + 0.5f);
	}
	__syncthreads();


	int wxb = patch_radius; // window x begin 
	int wyb = patch_radius; // window y begin
	int wxe = local_block_w - patch_radius; // window x end
	int wye = local_block_h - patch_radius; // window y end

	// go through window
	for (int wy = wyb; wy <= wye; wy++){
		for (int wx = wxb; wx <= wxe; wx++){

			float distance(0);
			// go through each patch
			for (int py = -patch_radius; py <= patch_radius; py++)
			for (int px = -patch_radius; px <= patch_radius; px++){

				// Main patch
				int px1 = threadIdx.x + window_radius + patch_radius + px;//fminf(x_size - 1, fmaxf(0, xId + px));
				int py1 = threadIdx.y + window_radius + patch_radius + py;//fminf(y_size - 1, fmaxf(0, yId + py));

				int px2 = wx + px;
				int py2 = wy + py;

				float tmp = LocalBlock[(px1)+(py1)*local_block_h] - LocalBlock[px2 + py2*local_block_w];
				distance += tmp*tmp*filter[center + px] * filter[center + py];


			}// go through each patch

			float w = exp(-distance * inv_sqr_sigma);
			new_value += w*LocalBlock[wx + wy*local_block_w];
			normalizer += w;
		}
	}// go through window

	d_out[yId*x_size + xId] = new_value / normalizer;
}

CMatrix<float> nonLocalMeanCudaShared(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma){

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
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float)* image_size));
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float)* image_size));
	checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float)* (2 * patch_radius + 1)));

	// Copy to the device
	checkCudaErrors(cudaMemcpy(d_image, image.data(), sizeof(float)* image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)* (2 * patch_radius + 1), cudaMemcpyHostToDevice));

	cudaBindTexture2D(0, texRef, d_image, image.xSize(), image.ySize(), sizeof(float)*image.xSize());

	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModePoint;
	texRef.normalized = false;

	// Compute kernel
	dim3 block(16, 16, 1);
	dim3 grid;
	grid.x = DIV_UP(image.xSize(), (float)block.x);
	grid.y = DIV_UP(image.ySize(), (float)block.y);
	int LocalBlockSize = (block.x + 2 * patch_radius + 2 * window_radius)*(block.y + 2 * patch_radius + 2 * window_radius)* sizeof(float);
	nlmKernelShared << < grid, block, LocalBlockSize >> >(d_out, d_filter, image.xSize(), image.ySize(), window_radius, patch_radius, inv_sqr_sigma);

	// Copy result from the device
	checkCudaErrors(cudaMemcpy((void*)result.data(), (void*)d_out, sizeof(float)* result.size(), cudaMemcpyDeviceToHost));

	//cudaDeviceSynchronize(); 
	//checkCudaErrors(cudaGetLastError());
	cudaFree(d_image);
	cudaFree(d_out);
	cudaFree(d_filter);
	cudaUnbindTexture(texRef);
	checkCudaErrors(cudaDeviceReset());
	return result;
}
