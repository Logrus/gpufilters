#pragma once
#ifndef NLMF_H_
#define NLMF_H_
#include <cstdio>
#include <helpers/CMatrix.h>

CMatrix<float> nonLocalMeanCudaNaiveTexture(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma);
CMatrix<float> nonLocalMeanCudaNaive(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma);

#endif // NLMF_H_