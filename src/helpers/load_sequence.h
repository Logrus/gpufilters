#ifndef LOAD_SEQUENCE_H_
#define LOAD_SEQUENCE_H_
#include <helpers/CVector.h>
#include <helpers/CMatrix.h>
#include <string>

CVector< CMatrix<float> > loadSequence(std::string Filename);

#endif