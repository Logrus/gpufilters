#ifndef AUXCU_H_
#define AUXCU_H_

#include <iomanip>
#include "CMatrix.h"
#include <vector>
#include <assert.h>

#ifndef CLIP
#define CLIP(min, val, max) (fminf(max, fmaxf(min,val)))
#endif

#ifndef DIV_UP
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

template <class T>
void displayDevMat(T* mat, int x_size, int y_size, int ystart = 0, int yend = 20, int xstart = 0, int xend = 31){
	CMatrix<T> res(x_size, y_size);
	checkCudaErrors(cudaMemcpy(res.data(), mat, sizeof(T) * res.size(), cudaMemcpyDeviceToHost));
	std::cout << "Matrix\n";
	for (int j = ystart; j < yend; j++)
		for (int i = xstart; i < xend; i++){
			std::cout << std::setw(3) << std::setprecision(2) << res(i, j) << " ";
			if (i == xend - 1)
				std::cout << std::endl;
		}

}

//void assertLists(std::vector<point> l1, std::vector<point> l2){
//	bool is_equal = false;
//	std::sort(l1.begin(), l1.end());
//	std::sort(l2.begin(), l2.end());
//	if (l1.size() < l2.size()){
//		for (int i = 0; i < l1.size(); i++)
//			if (l1[i] == l2[i]) {
//				is_equal = true;
//				break;
//			}
//	}
//	else{
//		for (int j = 0; j < l2.size(); j++)
//			if (l1[j] == l2[j]) {
//				is_equal = true;
//				break;
//			}
//	}
//	assert(!is_equal);
//}

#endif /* AUXCU_H */