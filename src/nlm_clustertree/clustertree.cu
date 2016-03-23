#include "clustertree.h"
#ifndef __linux__
#include <iso646.h>
#include <helpers/ImageDisplay.h>
#endif
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <algorithm>

#ifdef _DEBUG
ImageDisplay::ImageDisplay Display;
#endif

__global__ void _distacesToAll(float* d_image,
					   		   float* d_out,
					   		   float* filter,
							   int x_size,
							   int y_size,
					   		   float* mu,
							   unsigned int* indicator,
							   int node_index,
					   		   int patch_radius,
					   		   float inv_sqr_sigma){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if(indicator[yId*x_size + xId] != node_index){
		d_out[yId*x_size + xId] = -1;
		return;
	}

	int center = patch_radius;
	int vec_x_size = 2 * patch_radius + 1;

	float distance(0);
	// go through each patch
	for (int py = -patch_radius; py <= patch_radius; py++)
		for (int px = -patch_radius; px <= patch_radius; px++){

			// Main patch
			int px1 = CLIP(0, xId + px, x_size - 1);
			int py1 = CLIP(0, yId + py, y_size - 1);

			// Patch in the window
			//int px2 = CLIP(0, xpos + px, x_size - 1);
			//int py2 = CLIP(0, ypos + py, y_size - 1);

			float tmp = d_image[px1 + py1*x_size] - mu[(px + patch_radius) + (py + patch_radius)*vec_x_size];
			distance += tmp*tmp*filter[center + px] * filter[center + py];


		}// go through each patch

	float w = exp(-distance * inv_sqr_sigma);
	__syncthreads();
	d_out[yId*x_size + xId] = w;
}


__global__ void _compareTwoMatrixes(float *first_mat, 
	float *second_mat, 
	float *d_out, 
	unsigned int* indicator, 
	int node_index, 
	int x_size, 
	int y_size,
	int strictly){

	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	d_out[yId*x_size + xId] = 0;

	if (indicator[yId*x_size + xId] == node_index){
		// Read values
		float val1 = first_mat[yId*x_size + xId];
		float val2 = second_mat[yId*x_size + xId];
		if (strictly){
			if (val1>val2)
				d_out[yId*x_size + xId] = 1;
		}
		else{
			if (val1 >= val2)
				d_out[yId*x_size + xId] = 1;
		}
	}

}


// Updates global indicator mat based on local step matrix
__global__ void _updateIndicatorMatrix(unsigned int* d_indicator, int x_size, int y_size, float* d_step_indicator, int index1, int index2){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if (d_step_indicator[yId*x_size + xId] == 1)
		d_indicator[yId*x_size + xId] = index1;
	if (d_step_indicator[yId*x_size + xId] == 0)
		d_indicator[yId*x_size + xId] = index2;
}

__global__ void _matIntToFloat(unsigned int *d_in, float* d_out, int x_size, int y_size){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	d_out[yId*x_size + xId] = (float)d_in[yId*x_size + xId];
}

__global__ void _shiftMat(float* d_shifted, float* d_image, int x_size, int y_size, int x_shift, int y_shift, float* d_indicator){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if (!d_indicator[yId*x_size + xId]){
		d_shifted[yId*x_size + xId] = 0; // There could be smth left from previous calls
		return;
	}

	int x = CLIP(0, xId + x_shift, x_size - 1);
	int y = CLIP(0, yId + y_shift, y_size - 1);

	d_shifted[yId*x_size + xId] = d_image[y*x_size + x];

}

__global__ void _updateIndicator(unsigned int* d_indicator, int x_size, int y_size, float* d_mask, int index){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if (d_mask[yId*x_size + xId] == 1)
		d_indicator[yId*x_size + xId] = index;

}

__global__ void _getTheHeck(int2* list, unsigned int* d_indicator, int x_size, int y_size, int index){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if (d_indicator[yId*x_size + xId] != index){
		int2 p;
		p.x = -1;
		p.y = -1;
		list[yId*x_size + xId] = p;
	}
	else{

		int2 p;
		p.x = xId;
		p.y = yId;

		list[yId*x_size + xId] = p;
	}
}

__global__ void _fillRefMat(int** refmat, unsigned int* d_indicator, int x_size, int y_size, int* reference, int index){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	if (d_indicator[yId*x_size + xId] == index)
		refmat[yId*x_size + xId] = reference;
}

__global__ void _nlmCudaList(float* d_image, float* d_out, float* filter, int x_size, int y_size, int** refmat, int patch_radius, float inv_sqr_sigma, int w){
	int xId = blockDim.x * blockIdx.x + threadIdx.x;
	int yId = blockDim.y * blockIdx.y + threadIdx.y;

	// Check global image boundaries
	if (xId >= x_size || yId >= y_size)
		return;

	int center = patch_radius;
	float new_value(0); float normalizer(0);
	//int w = 1;
	for (int nx = -w; nx <= w; nx++)
		for (int ny = -w; ny <= w; ny++)
		{
			if ((ny + yId) >= 0 && (ny + yId) < y_size) // we shouldn't fall over the edge of the disc world
				if ((nx + xId) >= 0 && (nx + xId) < x_size)
					// Does it make sense to peek there at all? However we need to compute patch with it's corresponding list
					if ((refmat[(yId + ny)*x_size + (xId + nx)] != refmat[(yId)*x_size + (xId)]) || (nx==0 && ny==0))
					{ 
					int* list = refmat[(yId + ny)*x_size + (xId + nx)];
					int numel = list[0];

					// go through list of pixels
					for (int k = 1; k < numel; k++){

						float distance(0);
						// go through each patch
						for (int py = -patch_radius; py <= patch_radius; py++)
							for (int px = -patch_radius; px <= patch_radius; px++){

								// Main patch
								int px1 = CLIP(0, xId + px, x_size - 1);
								int py1 = CLIP(0, yId + py, y_size - 1);

								// Patch in the window
								int px2 = CLIP(0, list[k] + px, x_size - 1);
								int py2 = CLIP(0, list[k + numel] + py, y_size - 1);

								float tmp = d_image[px1 + py1*x_size] - d_image[px2 + py2*x_size];
								distance += tmp*tmp*filter[center + px] * filter[center + py];

							}

						float w = exp(-distance * inv_sqr_sigma);
						new_value += w*d_image[list[k] + list[k + numel] * x_size];
						normalizer += w;
					} // Go through list with pixels
			} // Makes sense if we are not out
	}// Go through neig
	

	// We need syncthreads before writing the final result
	//__syncthreads();
	d_out[yId*x_size + xId] = new_value / normalizer;
}


ClusterTree::ClusterTree(CMatrix<float> data, int patch_radius, float sqr_sigma) 
	: x_size(data.xSize()), y_size(data.ySize()), patch_radius(patch_radius), sqr_sigma(sqr_sigma), image(data){
	
	this->inv_sqr_sigma = 1.0f / sqr_sigma;

	// Allocate memory for the reference matrix
	//this->refmat = new node*[this->x_size*this->y_size];
	

	this->Head = new node;
	this->Head->index = 0;
	this->Head->leaf = 1; // this could be our only node so for now it's a leaf
	this->Head->left = NULL; // those pointers shouldn't point anywhere
	this->Head->right = NULL; // for now
	// Initialize domain list
	for (int j = 0; j < y_size; ++j)
		for (int i = 0; i < x_size; ++i){
			point p;
			p.x = i;
			p.y = j;
			this->Head->coordinates.push_back(p);
		}
	node_count = 0; //initialize variable (head excluded)

	allocateIdicatorMatrix();

	mu_x_size = (2 * patch_radius + 1);

	// Device initialization
	this->block = dim3(32, 32, 1);
	this->grid.x = DIV_UP(image.xSize(), (float)block.x);
	this->grid.y = DIV_UP(image.ySize(), (float)block.y);

	// create a gauss lut for 1D
	this->h_filter = new float[mu_x_size];
	float* center = h_filter + patch_radius;
	for (int x = -patch_radius; x <= patch_radius; ++x)
		*(center + x) = std::exp(-0.5*x*x / (patch_radius*patch_radius));
	
	this->image_size = data.xSize()*data.ySize();
	this->maxlvl = 1;
	this->maxpoints = 50;
	this->w = 1;
	// Initialize a random generator
#ifdef _DEBUG
	srand(5);
#else
	srand(time(NULL));
#endif

	// Allocate memory on the device
	checkCudaErrors(cudaMalloc((void ***)&this->refmat, sizeof(int*) * image_size));

	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(float) * image_size));

	checkCudaErrors(cudaMalloc((void**)&d_mat1, sizeof(float) * image_size));
	checkCudaErrors(cudaMemset(d_mat1, 0, sizeof(float) * image_size));

	checkCudaErrors(cudaMalloc((void**)&d_mat2, sizeof(float) * image_size));
	checkCudaErrors(cudaMemset(d_mat2, 0, sizeof(float) * image_size));

	checkCudaErrors(cudaMalloc((void**)&d_step_indicator, sizeof(float) * image_size));
	checkCudaErrors(cudaMemset(d_step_indicator, 0, sizeof(float) * image_size));

	checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float) * (2 * patch_radius + 1)));

	// Copy to the device
	checkCudaErrors(cudaMemcpy(d_image, image.data(), sizeof(float) * image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * (mu_x_size), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_mu1, sizeof(float) * SQR(2 * patch_radius + 1)));
	checkCudaErrors(cudaMalloc((void**)&d_mu2, sizeof(float) * SQR(2 * patch_radius + 1)));

	checkCudaErrors(cudaMalloc(&d_shifted, sizeof(float) * image_size));

};

void ClusterTree::setMaximumLevel(int lvl){
	if (lvl > 0)
		this->maxlvl = lvl;
	else{
		std::cerr << "Max level could be between 0 or 10" << std::endl;
		if (lvl <= 0) { lvl = 1; std::cout << "Max level has been set to 1"; };
		if (lvl > 10) { lvl = 10; std::cout << "Max level has been set to 10"; };
	}
}

void ClusterTree::setMaximumOfPointsInTheLeafs(int points){
	if (points > 2)
		this->maxpoints = points;
	else{
		std::cerr << "Max point in leaf should be more than 2" << std::endl;
	}
}

ClusterTree::~ClusterTree(){
	// Everything allocated should be deleted
	destroyTree(this->Head);
	//destroyIndicatorMatrix();
	//delete[] this->refmat;

	cudaFree(d_image);
	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_mu1);
	cudaFree(d_mu2);
	cudaFree(d_shifted);
	cudaFree(d_step_indicator);
	cudaFree(d_filter);
	cudaFree(d_indicatorMat);
	//for (int i = 0; i < image_size; i++){
	//	if (refmat[i])
	//		cudaFree(refmat[i]);
	//}
	cudaFree(refmat);
	//cudaFree(d_indicator);
	checkCudaErrors(cudaDeviceReset());
}

void ClusterTree::displayCoordinateList(node* n){
	for (int i = 0; i < n->coordinates.size(); ++i){
		std::cout << "x: " << n->coordinates[i].x << "y: " << n->coordinates[i].y << std::endl;
	}
}

node* ClusterTree::getHead(){
	return this->Head;
};

int ClusterTree::getIndexOfNode(node* n){
	return n->index;
}

unsigned int* ClusterTree::getIndicatorMatrix(){
	if (this->d_indicatorMat == NULL)
		this->allocateIdicatorMatrix();
	return this->d_indicatorMat;
}

void ClusterTree::destroyTree(node* p){
	if (p->left){ // Does it have left node?
		this->destroyTree(p->left); // Traverse further
		p->left = NULL;
	}
	if (p->right){ // Does it have right node?
		this->destroyTree(p->right); // Traverse further
		p->right = NULL;
	}
	// Doesn't have childs but still considered non leaf
	if (!p->right && !p->left && !p->leaf){
		p->leaf = 1;
	}
	if (p->leaf){ // Is this a leaf node? (shouldn't have left\right)
		//delete[] p->mu; // Kill it's allocated memory
		delete p; // Kill it
	}
	
	
}

point ClusterTree::sampleRand(node* n){
	unsigned int minx = 0;
	unsigned int maxx = n->coordinates.size()-1;
	unsigned int index = rand() % (maxx - minx) + minx; // random value in [min, max]
	point p;
	p.x = n->coordinates[index].x;
	p.y = n->coordinates[index].y;
//#ifdef _DEBUG
//	std::cout << "xval: " << n->coordinates[index].x << std::endl;
//	std::cout << "yval: " << n->coordinates[index].y << std::endl;
//#endif
	return p;
}

void ClusterTree::allocateIdicatorMatrix(){
	int size = this->x_size*this->y_size;
	unsigned int* initmat = new unsigned int[size];
	std::fill(initmat, initmat + size, 0); // Fill with initial values [Head node index is 0]

	// Allocate memory
	checkCudaErrors(cudaMalloc(&this->d_indicatorMat, size*sizeof(unsigned int)));

	// Copy
	checkCudaErrors(cudaMemcpy(this->d_indicatorMat, initmat, size*sizeof(unsigned int), cudaMemcpyHostToDevice));

}

void ClusterTree::destroyIndicatorMatrix(){
	if (this->d_indicatorMat != NULL)
		checkCudaErrors(cudaFree(this->d_indicatorMat));
	this->d_indicatorMat = NULL;
}

void ClusterTree::create2Childs(node* current){
	// Allocate memory for new nodes
	node* left = new node;
	node* right = new node;

	// A proper initialization of nodes
	left->leaf = 1;
	left->left = NULL; left->right = NULL;
	//left->mu = new float[patch_radius * 2 + 1];
	left->index = ++node_count;
	left->parent = current;

	right->leaf = 1;
	right->left = NULL; right->right = NULL;
	//right->mu = new float[patch_radius * 2 + 1];
	right->index = ++node_count;
	right->parent = current;

	current->leaf = 0; // This is no longer a leaf node
	current->left = left;
	current->right = right;

}
void ClusterTree::setW(int num){
	if (num < 0) return;
	this->w = num;
}

int ClusterTree::getMaximumLevel(){
	return this->maxlvl;
}
int ClusterTree::getMaximumOfPointsInTheLeafs(){
	return this->maxpoints;
}
int ClusterTree::getW(){
	return this->w;
}

float* ClusterTree::pickVecByPos(float* image, point p, int x_size, int y_size, int patch_radius){
	int vecsize = SQR(2 * patch_radius + 1);
	int vs_half = vecsize / 2;
	int vec_x_size = 2 * patch_radius + 1;
	float* mu = new float[vecsize];
	for (int i = 0; i < vecsize; ++i) mu[i] = 0;
	for (int py = -patch_radius; py <= patch_radius; py++)
		for (int px = -patch_radius; px <= patch_radius; px++){
			int px1 = CLIP(0, px + p.x, x_size - 1);
			int py1 = CLIP(0, py + p.y, y_size - 1);
			mu[(py + patch_radius)*vec_x_size + (px + patch_radius)] += image[py1*x_size + px1];
		}
	return mu;
}

struct nonneg{
	__host__ __device__
	bool operator()(const int2 x)
	{
		return (x.x != -1);
	}
};

void ClusterTree::updateCoordinateList(node* n){

	int2* d_list;
	checkCudaErrors(cudaMalloc(&d_list, sizeof(int2) * image_size));
	checkCudaErrors(cudaMemset(d_list, 0, sizeof(int2) * image_size));
	_getTheHeck << <grid, block >> >(d_list, d_indicatorMat, x_size, y_size, n->index);


	thrust::device_ptr<int2> d_list_(d_list);
	int size = image_size; //DIV_UP(image_size, divisor);
	int2* result = new int2[size];
	thrust::device_vector<int2> d_vec(size);
	thrust::host_vector<int2> h_vec(size);
	
	thrust::copy_if(d_list_, d_list_ + size, d_vec.begin(), nonneg());

	h_vec = d_vec;
	thrust::copy(h_vec.begin(), h_vec.end(), result);

	n->coordinates.clear();

	for (int i = 0;  ; ++i){
		if (i == 0 && !result[i].x && !result[i].y) { // The first point could be 0 0 for some nodes
			point p; p.x = result[i].x; p.y = result[i].y;
			n->coordinates.push_back(p);
			continue;
		}
		if (result[i].x <= 0 && result[i].y <= 0)
			break;
		point p;
		p.x = result[i].x; p.y = result[i].y;
		n->coordinates.push_back(p);
//#ifdef _DEBUG
//		std::cout << "Result x " << result[i].x << " y " << result[i].y << std::endl;
//		std::cout << i << std::endl;
//#endif
	}
	delete[] result;
	checkCudaErrors(cudaFree(d_list));
}

// This function computes distortion measure J from mat1, mat2, d_out
// Trat no time to write it
//float ClusterTree::getDistortionMeasure(){
//	return 0;
//}

int counter(0);

void ClusterTree::buildTree(node* n, int level){


#ifdef _DEBUG
	CMatrix<float> result(x_size, y_size);
#endif

	level++;
	node* current = n;
	unsigned int* d_indicator = getIndicatorMatrix();

	if (current->coordinates.size() < maxpoints || level > maxlvl){ //
		int* d_link, *xarr, *yarr;
		int size = current->coordinates.size();
		std::sort(current->coordinates.begin(), current->coordinates.end());
		xarr = new int[size+1];
		yarr = new int[size];
		xarr[0] = size;
 		for (int i = 0; i < size; i++){
			xarr[i+1] = current->coordinates[i].x;
			yarr[i] = current->coordinates[i].y;
		}
		checkCudaErrors(cudaMalloc(&d_link, sizeof(int) * (2*size+1)));
		//checkCudaErrors(cudaMemset(d_link, size, sizeof(int)+1));
		checkCudaErrors(cudaMemcpy(d_link, xarr, sizeof(int) * (size+1), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_link+ 1 + size, yarr, sizeof(int) * size, cudaMemcpyHostToDevice));
		_fillRefMat<<<grid, block>>>(this->refmat, d_indicator, x_size, y_size, d_link, current->index);
		delete[] xarr;
		delete[] yarr;
		//std::cout << "Dumped " << size << " points" << std::endl;
		current->coordinates.clear();
		return;
	}
	

	point p1, p2; // define and initialize variables
	// Sample two random positions in the image
	p1 = sampleRand(current);
	p2 = sampleRand(current);

	// Initialize mu vectors
	float *mu1 = pickVecByPos(image.data(), p1, x_size, y_size, patch_radius);
	float *mu2 = pickVecByPos(image.data(), p2, x_size, y_size, patch_radius);
	
	
	
	double mu1_normalizer, mu2_normalizer;

	cudaStream_t s1;
	cudaStream_t s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	// ==========================
	// K-Means steps
	for (int k = 0; k < 5; ++k){
		checkCudaErrors(cudaMemset(d_step_indicator, 0, sizeof(float) * image_size)); // Reset the function

		// Update values on the device
		checkCudaErrors(cudaMemcpyAsync(d_mu1, mu1, sizeof(float) * SQR(2 * patch_radius + 1), cudaMemcpyHostToDevice, s1));
		checkCudaErrors(cudaMemcpyAsync(d_mu2, mu2, sizeof(float) * SQR(2 * patch_radius + 1), cudaMemcpyHostToDevice, s2));

		
		// Assign step
		_distacesToAll << < grid, block, 0, s1 >> >(d_image, d_mat1, d_filter, x_size, y_size, d_mu1, d_indicator, current->index, patch_radius, inv_sqr_sigma);
		_distacesToAll << < grid, block, 0, s2 >> >(d_image, d_mat2, d_filter, x_size, y_size, d_mu2, d_indicator, current->index, patch_radius, inv_sqr_sigma);
		_compareTwoMatrixes << < grid, block >> >(d_mat1, d_mat2, d_step_indicator, d_indicator, current->index, x_size, y_size, 1);

		//checkCudaErrors(cudaMemcpy((void*)result.data(), d_step_indicator, sizeof(float) * result.size(), cudaMemcpyDeviceToHost));
		////result.writeToTXT("res.txt");
		//result.normalize(0, 255);
		//Display.Display(result, "res");

		thrust::device_ptr<float> d_step_indicator_(d_step_indicator);
		// Update step
		// Compute mu1 through reduction
		mu1_normalizer = thrust::reduce(d_step_indicator_, d_step_indicator_ + image_size);
		if (mu1_normalizer == 0){

			int* d_link, *xarr, *yarr;
			int size = current->coordinates.size();
			std::sort(current->coordinates.begin(), current->coordinates.end());
			xarr = new int[size + 1];
			yarr = new int[size];
			xarr[0] = size;
			for (int i = 0; i < size; i++){
				xarr[i + 1] = current->coordinates[i].x;
				yarr[i] = current->coordinates[i].y;
			}
			checkCudaErrors(cudaMalloc(&d_link, sizeof(int) * (2 * size + 1)));
			//checkCudaErrors(cudaMemset(d_link, size, sizeof(int)+1));
			checkCudaErrors(cudaMemcpy(d_link, xarr, sizeof(int) * (size + 1), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_link + 1 + size, yarr, sizeof(int) * size, cudaMemcpyHostToDevice));
			_fillRefMat << <grid, block >> >(this->refmat, d_indicator, x_size, y_size, d_link, current->index);
			delete[] xarr;
			delete[] yarr;
			//std::cout << "Dumped " << size << " points" << std::endl;
			current->coordinates.clear();

			return;
		}

//#pragma omp parallel for
		for (int j = -patch_radius; j <= patch_radius; j++)
			for (int i = -patch_radius; i <= patch_radius; i++){
				//int sh = (patch_radius + j) * image_size;
				_shiftMat << <grid, block >> >(d_shifted, d_image, x_size, y_size, i, j, d_step_indicator);
				thrust::device_ptr<float> d_shifted_(d_shifted);
				double val = thrust::reduce(d_shifted_, d_shifted_ + image_size);
				mu1[(patch_radius + i) + (2 * patch_radius + 1)*(patch_radius + j)] = val / mu1_normalizer;
			}

		// Compute mu2 through reduction
		_compareTwoMatrixes << < grid, block >> >(d_mat2, d_mat1, d_step_indicator, d_indicator, current->index, x_size, y_size, 0);
		mu2_normalizer = thrust::reduce(d_step_indicator_, d_step_indicator_ + image_size);
		if (mu2_normalizer == 0){

			int* d_link, *xarr, *yarr;
			int size = current->coordinates.size();
			std::sort(current->coordinates.begin(), current->coordinates.end());
			xarr = new int[size + 1];
			yarr = new int[size];
			xarr[0] = size;
			for (int i = 0; i < size; i++){
				xarr[i + 1] = current->coordinates[i].x;
				yarr[i] = current->coordinates[i].y;
			}
			checkCudaErrors(cudaMalloc(&d_link, sizeof(int) * (2 * size + 1)));
			//checkCudaErrors(cudaMemset(d_link, size, sizeof(int)+1));
			checkCudaErrors(cudaMemcpy(d_link, xarr, sizeof(int) * (size + 1), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_link + 1 + size, yarr, sizeof(int) * size, cudaMemcpyHostToDevice));
			_fillRefMat << <grid, block >> >(this->refmat, d_indicator, x_size, y_size, d_link, current->index);
			delete[] xarr;
			delete[] yarr;
			//std::cout << "Dumped " << size << " points" << std::endl;
			current->coordinates.clear();

			return;
		}

//#pragma omp parallel for
		for (int j = -patch_radius; j <= patch_radius; j++)
			for (int i = -patch_radius; i <= patch_radius; i++){
				//int sh = (patch_radius + j) * image_size;
				_shiftMat << <grid, block >> >(d_shifted, d_image, x_size, y_size, i, j, d_step_indicator);
				thrust::device_ptr<float> d_shifted_(d_shifted);
				double val = thrust::reduce(d_shifted_, d_shifted_ + image_size);
				mu2[(patch_radius + i) + (2 * patch_radius + 1)*(patch_radius + j)] = val / mu2_normalizer;
			}

#ifdef _DEBUG
		assert((mu1_normalizer + mu2_normalizer == (x_size*y_size)) || level > 1);
#endif

	}


#ifdef _DEBUG
	if ((counter % 1) == 0){
		// tmp display step
		float* tmp;
		checkCudaErrors(cudaMalloc(&tmp, sizeof(float) * image_size));
		checkCudaErrors(cudaMemset(tmp, 0, sizeof(float) * image_size));
		_matIntToFloat << <grid, block >> >(d_indicator, tmp, x_size, y_size);
		checkCudaErrors(cudaMemcpy((void*)result.data(), tmp, sizeof(float) * result.size(), cudaMemcpyDeviceToHost));
		//result.writeToTXT("res.txt");
		result.normalize(0, 255);
		Display.Display(result, "res");
		result.writeToPGM(SSTR("notes/images/kmeansfull/salesman" << counter << ".pgm").c_str());
	}
	counter++;
#endif

	// ==========================
	// Go to the next level

	// Allocate new nodes
	create2Childs(current);
	current->coordinates.clear(); // oh save some memory, please..
	// Update main indicator matrix
	// Update main indicator matrix
	_compareTwoMatrixes << < grid, block, 0, s1 >> >(d_mat1, d_mat2, d_step_indicator, d_indicator, current->index, x_size, y_size, 1);
	_updateIndicator << <grid, block, 0, s1 >> >(d_indicator, x_size, y_size, d_step_indicator, current->left->index);
	_compareTwoMatrixes << < grid, block, 0, s2 >> >(d_mat2, d_mat1, d_step_indicator, d_indicator, current->index, x_size, y_size, 0);
	_updateIndicator << <grid, block, 0, s2 >> >(d_indicator, x_size, y_size, d_step_indicator, current->right->index);
	
	// Assign them right pixels
	updateCoordinateList(current->right);
	updateCoordinateList(current->left);

#ifdef _DEBUG
	// Check sizes of lists
	int ss1 = current->left->coordinates.size();
	assert(mu1_normalizer == ss1); 
	int ss2 = current->right->coordinates.size();
	assert(mu2_normalizer == ss2);
	std::cout << ss1 + ss2 << " " << x_size*y_size << std::endl;
	assert(((ss1 + ss2) == x_size*y_size) || level > 1);
#endif

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);

	// Declare foo as arrays of array 8 of pointer to pointer 
	// to function returning a pointer t array of pointer to char
	// and we need to go deeper
	if ( current->left != NULL)
			buildTree(current->left, level);
	if ( current->right != NULL)
			buildTree(current->right, level);

		//std::cout << node_count << std::endl;
};

CMatrix<float> ClusterTree::nlm(){
	CMatrix<float> result(x_size, y_size);
	float* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(float)*image_size));
	checkCudaErrors(cudaMemset(d_out, 0,  sizeof(float)*image_size));

	_nlmCudaList << < grid, block >> >(d_image, d_out, d_filter, x_size, y_size, refmat, patch_radius, inv_sqr_sigma, this->w);

	checkCudaErrors(cudaMemcpy(result.data(), d_out, sizeof(float)*image_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_out));
	return result;
}
