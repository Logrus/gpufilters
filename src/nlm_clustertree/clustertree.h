#ifndef CLUSTERTREE_H_
#define CLUSTERTREE_H_
#include <helpers/auxcu.h>
#include <helpers/utils.h>
#include <helpers/CMatrix.h>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <sstream>

struct point{
	unsigned int x, y;
	point() {};
	point (int a, int b) {
		x = a; y = b;
	}
	bool operator ==(const point &a){
		return (x == a.x && y == a.y);
	}

	bool operator >(const point &a){
		if (x != a.x)
			return (x > a.x);
		else
			return (y > a.y);
	}

	bool operator <(const point &a){
		if (x != a.x)
			return (x < a.x);
		else
			return (y < a.y);
	}

	bool operator >=(const point &a){
		if (x != a.x)
			return (x >= a.x);
		else
			return (y >= a.y);
	}

	bool operator <=(const point &a){
		if (x != a.x)
			return (x <= a.x);
		else
			return (y <= a.y);
	}
};

//bool sortPoints(const point &a, const point &b){
//		if (b.x != a.x)
//			return (b.x < a.x);
//		else
//			return (b.y < a.y);
//	}


struct node{
	int index;
	bool leaf;
	// lists with coordinates of pixels of this node
	// only leafs should have them
	std::vector< point > coordinates; 
	//float* mu;
	node *left, *right, *parent;
};

class ClusterTree {
//private: // switch all to public for debugging
public:
	// Host functions and vars
	CMatrix<float> image;
	node* Head; // We need to store a pointer to the head
	int maxlvl, maxpoints;
	int node_count;
	int w;
	int x_size, y_size, image_size;
	int patch_radius;
	float sqr_sigma, inv_sqr_sigma;
	int mu_x_size; // Common for all nodes in the tree
	//node** refmat; // Reference matrix	
	point sampleRand(node* /*n*/);// Sample random position in the domain of current node
	CMatrix<float> computeDistances(const CMatrix<float> &image, int xpos, int ypos);
	void compareMatrices(CMatrix<float> &mat1, CMatrix<float> &mat2);
	void updateRefmat(const CMatrix<float> &indicator); // Update refmat according to the indicator matrix
	void create2Childs(node* current);
	void destroyTree(node* p);
	unsigned int* returnCoordXArray(node* /*n*/, int& /*d_elem_size*/);
	unsigned int* returnCoordYArray(node* /*n*/, int& /*d_elem_size*/);
	void allocateIdicatorMatrix();
	void destroyIndicatorMatrix();
	void updateIndicatorMatrix(std::vector<point> points, int index); // TODO: probably delete
	void updateCoordinateList(node* n);
	void displayCoordinateList(node* n);
	float* pickVecByPos(float* image, point p, int x_size, int y_size, int patch_radius);
	// Pointers on the host
	float* h_filter;

	// Pointers to the device memory
	unsigned int* d_indicatorMat; // Integer indicator matrix on the device
	float *d_image, 
		  *d_filter, 
		  *d_step_indicator, // Comparison of mu1 and mu2 [Is computed in each kmeans step]
		  *d_mat1,  // Distances from mu1 
		  *d_mat2;  // Distances from mu2
	float *d_mu1, 
		  *d_mu2;
	float * d_shifted;
	int** refmat;
	// Device parameters
	dim3 block;
	dim3 grid;

	// Functions working with GPU
	float getDistortionMeasure();

public:
	ClusterTree(CMatrix<float> /*data*/, int /*patch_radius*/, float /*sqr_sigma*/);
	node* getHead(); // Return pointer to the root node
	int getIndexOfNode(node* n);
	unsigned int* getIndicatorMatrix(); // Return pointer to the indicator matrix on the device
	void buildTree(node* n, int level);
	void setMaximumLevel(int lvl);
	void setMaximumOfPointsInTheLeafs(int points);
	void setW(int num);
	int getMaximumLevel();
	int getMaximumOfPointsInTheLeafs();
	int getW();
	CMatrix<float> nlm();
	~ClusterTree();
};

void buidClusterTree(CMatrix<float> image, int patch_radius);
#endif /* CLUSTERTREE_H_ */