#pragma once
#ifndef NLMSH_H_
#define NLMSH_H_
#include <cstdio>
#include <helpers/CMatrix.h>

CMatrix<float> nonLocalMeanCudaShared(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma);


#endif // NLMSH_H_