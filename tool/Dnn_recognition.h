#pragma once
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include "atlimage.h"
using namespace std;

typedef void * cv_handle_t;
typedef struct ImageData {
	ImageData() {
		data = nullptr;
		width = 0;
		height = 0;
		num_channels = 0;
	}

	ImageData(int32_t img_width, int32_t img_height,
		int32_t img_num_channels = 1) {
		data = nullptr;
		width = img_width;
		height = img_height;
		num_channels = img_num_channels;
	}

	uint8_t* data;
	int32_t width;
	int32_t height;
	int32_t num_channels;
} ImageData;

//#define DNN_API __declspec(dllexport)
#define DNN_API 
class DNN_recogniton
{
public:
	
	DNN_API DNN_recogniton(string pca_mean="mean.dat", string pca_proj="proj.dat");
	DNN_API int DNN_getfeat(ImageData aligned_face, float* feat);
	DNN_API ~DNN_recogniton();
	

private:
	//DNN_API DNN_recogniton();
	cv_handle_t NN;
	//network<graph> nn;
	//cv::Size input_geometry_;
	int DNN_forward(ImageData aligned_face, float* feat_mat);
	string feat_layer_name_;
	float *feat_mat_;
	float *mean_feat;
	float **PCA_Proj;
	int num_channels_;
	int init_net();
	/*void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);*/

};
