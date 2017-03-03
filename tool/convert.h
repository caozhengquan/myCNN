#ifndef CONVERT_H
#define CONVERT_H
#include "atlimage.h"
#include "tiny_dnn/tiny_dnn.h"

#include "Dnn_recognition.h"




void ptr2vec_t(tiny_dnn::vec_t& in, const ImageData& im);
void cimage2vec_t(tiny_dnn::vec_t &in, CImage& im);
void cimage2ptr(unsigned char * ptr, CImage &im);
#endif