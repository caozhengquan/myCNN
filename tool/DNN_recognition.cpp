// DNN_recognition.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "tiny_dnn/tiny_dnn.h"
#include "Dnn_recognition.h"
#include "convert.h"
#include <stdio.h>
#include <fstream>
#define FEAT_NUM 256
#define FEAT_PCA_NUM 100
using namespace tiny_dnn;
void ptr2vec_t(vec_t& in, const ImageData& im);
//void cimage2vec_t(vec_t &in, CImage& im);
//DNN_recogniton::DNN_recogniton()
//{
//}

int DNN_recogniton::init_net()
{
	//Device my_gpu_device(device_t::CPU, 1, 0);

	auto in = std::make_shared<input_layer>(shape3d(96, 112, 3));
	convolutional_layer<>* c1 = new convolutional_layer<>(96, 112, 5, 3, 96, padding::same, true, 1, 1);
	cout << "c1: " << c1->in_channels() << "  " << c1->out_channels() << endl;
	slice_layer* sl1 =new slice_layer(shape3d(96, 112, 96), slice_type::slice_channels, 2); //shape3d(112, 96, 96)
	linear_layer<>* l11=new linear_layer<>(112 * 96 * 48);
	linear_layer<>* l12 = new linear_layer<>(112 * 96 * 48);
	elementwise_max_layer* max_elt1 = new elementwise_max_layer(2, 112 * 96 * 48);
	max_pooling_layer<>* pl1 =new max_pooling_layer<>(96, 112, 48, 2);

	convolutional_layer<>* c2 = new convolutional_layer<>(48, 56, 1, 48, 96, padding::same, true, 1, 1);
	slice_layer* sl2 = new slice_layer(shape3d(48, 56, 96), slice_type::slice_channels, 2);
	linear_layer<>* l21 = new linear_layer<>(56 * 48 * 48);
	linear_layer<>* l22 = new linear_layer<>(56 * 48 * 48);
	elementwise_max_layer* max_elt2 = new elementwise_max_layer(2, 56 * 48 * 48);


	convolutional_layer<>* c3 = new convolutional_layer<>(48, 56, 3, 48, 192, padding::same, true, 1, 1);
	slice_layer* sl3 = new slice_layer(shape3d(48, 56, 192), slice_type::slice_channels, 2);
	linear_layer<>* l31 = new linear_layer<>(56 * 48 * 96);
	linear_layer<>* l32 = new linear_layer<>(56 * 48 * 96);
	elementwise_max_layer* max_elt3 = new elementwise_max_layer(2, 56 * 48 * 96);
	max_pooling_layer<>* pl3 = new max_pooling_layer<>(48, 56, 96, 2);

	convolutional_layer<>* c4 = new convolutional_layer<>(24, 28, 1, 96, 192, padding::same, true, 1, 1);
	slice_layer* sl4 = new slice_layer(shape3d(24, 28, 192), slice_type::slice_channels, 2);
	linear_layer<>* l41 = new linear_layer<>(28 * 24 * 96);
	linear_layer<>* l42 = new linear_layer<>(28 * 24 * 96);
	elementwise_max_layer* max_elt4 = new elementwise_max_layer(2, 28 * 24 * 96);


	convolutional_layer<>* c5 = new convolutional_layer<>(24, 28, 3, 96, 384, padding::same, true, 1, 1);
	slice_layer* sl5 = new slice_layer(shape3d(24, 28, 384), slice_type::slice_channels, 2);
	linear_layer<>* l51 = new linear_layer<>(28 * 24 * 192);
	linear_layer<>* l52 = new linear_layer<>(28 * 24 * 192);
	elementwise_max_layer* max_elt5 = new elementwise_max_layer(2, 28 * 24 * 192);
	max_pooling_layer<>* pl5 = new max_pooling_layer<>(24, 28, 192, 2);

	convolutional_layer<>* c6 = new convolutional_layer<>(12, 14, 1, 192, 384, padding::same, true, 1, 1);
	slice_layer* sl6 = new slice_layer(shape3d(12, 14, 384), slice_type::slice_channels, 2);
	linear_layer<>* l61 = new linear_layer<>(14 * 12 * 192);
	linear_layer<>* l62 = new linear_layer<>(14 * 12 * 192);
	elementwise_max_layer* max_elt6 = new elementwise_max_layer(2, 14 * 12 * 192);


	convolutional_layer<>* c7 = new convolutional_layer<>(12, 14, 3, 192, 256, padding::same, true, 1, 1);
	slice_layer* sl7 = new slice_layer(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<>* l71 = new linear_layer<>(14 * 12 * 128);
	linear_layer<>* l72 = new linear_layer<>(14 * 12 * 128);
	elementwise_max_layer* max_elt7 = new elementwise_max_layer(2, 14 * 12 * 128);


	convolutional_layer<>* c8 = new convolutional_layer<>(12, 14, 1, 128, 256, padding::same, true, 1, 1);
	slice_layer* sl8 = new slice_layer(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<>* l81 = new linear_layer<>(14 * 12 * 128);
	linear_layer<>* l82 = new linear_layer<>(14 * 12 * 128);
	elementwise_max_layer* max_elt8 = new elementwise_max_layer(2, 14 * 12 * 128);


	convolutional_layer<>* c9 = new convolutional_layer<>(12, 14, 3, 128, 256, padding::same, true, 1, 1);
	slice_layer* sl9 = new slice_layer(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<>* l91 = new linear_layer<>(14 * 12 * 128);
	linear_layer<>* l92 = new linear_layer<>(14 * 12 * 128);
	elementwise_max_layer* max_elt9 = new elementwise_max_layer(2, 14 * 12 * 128);
	max_pooling_layer<>* pl9 = new max_pooling_layer<>(12, 14, 128, 2);

	fully_connected_layer<>* fc10=new fully_connected_layer<>(7 * 6 * 128, 512);
	slice_layer* sl10 = new slice_layer(shape3d(1, 1, 512), slice_type::slice_channels, 2);
	linear_layer<>* l101 = new linear_layer<>(1 * 1 * 256);
	linear_layer<>* l102 = new linear_layer<>(1 * 1 * 256);
	elementwise_max_layer* max_elt10 =new elementwise_max_layer(2, 1 * 1 * 256);

#ifdef USE_OPENCL
	my_gpu_device.registerOp(*c1);
	my_gpu_device.registerOp(*c2);
#endif
	
	network<graph>* nn=new network<graph>();
	NN = (cv_handle_t)(nn);
	// sl1 << (l11 , l12) ;
	*c1 << *sl1 << (*l11, *l12) << *max_elt1;
	cout << "construct 1" << endl;
	*max_elt1 << *pl1 << *c2 << *sl2 << (*l21, *l22) << *max_elt2;
	cout << "construct 2" << endl;
	*max_elt2 << *c3 << *sl3 << (*l31, *l32) << *max_elt3;
	cout << "construct 3" << endl;
	*max_elt3 << *pl3 << *c4 << *sl4 << (*l41,* l42) << *max_elt4;
	cout << "construct 4" << endl;
	*max_elt4 <<* c5 <<* sl5 << (*l51, *l52) << *max_elt5;
	cout << "construct 5" << endl;
	*max_elt5 << *pl5 << *c6 << *sl6 << (*l61, *l62) << *max_elt6;
	cout << "construct 6" << endl;
	*max_elt6 << *c7 << *sl7 << (*l71, *l72) << *max_elt7;
	cout << "construct 7" << endl;
	*max_elt7 << *c8 << *sl8 << (*l81, *l82) <<* max_elt8;
	cout << "construct 8" << endl;
	*max_elt8 << *c9 << *sl9 << (*l91, *l92) <<* max_elt9;
	cout << "construct 9" << endl;
	*max_elt9 << *pl9 << *fc10 << *sl10 << (*l101, *l102) << *max_elt10;
	construct_graph(*nn, { c1 }, { max_elt10 });
	cout << "construct finish" << endl;
	cout <<" nn->in_data_size() "<< nn->in_data_size() << endl;
	cout << nn << " == " << NN << endl;
	load_weights_conv_fromfile(".\\weight\\conv1.dat", c1);
	load_weights_conv_fromfile(".\\weight\\conv2a.dat", c2);
	load_weights_conv_fromfile(".\\weight\\conv2.dat", c3);
	load_weights_conv_fromfile(".\\weight\\conv3a.dat", c4);
	load_weights_conv_fromfile(".\\weight\\conv3.dat", c5);
	load_weights_conv_fromfile(".\\weight\\conv4a.dat", c6);
	load_weights_conv_fromfile(".\\weight\\conv4.dat", c7);
	load_weights_conv_fromfile(".\\weight\\conv5a.dat", c8);
	load_weights_conv_fromfile(".\\weight\\conv5.dat", c9);
	load_weights_fullyconnected_fromfile(".\\weight\\fc1.dat", fc10);
	return 1;
}

//int DNN_recogniton::init_net()
//{
//	Device my_gpu_device(device_t::CPU, 1, 0);
//
//	auto in = std::make_shared<input_layer>(shape3d(96, 112, 3));
//	convolutional_layer<> c1(96, 112, 5, 3, 96, padding::same, true, 1, 1
//#ifndef USE_OPENCL 
//		);
//#else
//		, core::backend_t::opencl);
//#endif
//	cout << "c1: " << c1.in_channels() << "  " << c1.out_channels() << endl;
//	slice_layer sl1(shape3d(96, 112, 96), slice_type::slice_channels, 2); //shape3d(112, 96, 96)
//	linear_layer<> l11(112 * 96 * 48);
//	linear_layer<> l12(112 * 96 * 48);
//	elementwise_max_layer max_elt1(2, 112 * 96 * 48);
//	max_pooling_layer<> pl1(96, 112, 48, 2);
//
//	convolutional_layer<> c2(48, 56, 1, 48, 96, padding::same, true, 1, 1
//#ifndef USE_OPENCL 
//		);
//#else
//		, core::backend_t::opencl);
//#endif
//	slice_layer sl2(shape3d(48, 56, 96), slice_type::slice_channels, 2);
//	linear_layer<> l21(56 * 48 * 48);
//	linear_layer<> l22(56 * 48 * 48);
//	elementwise_max_layer max_elt2(2, 56 * 48 * 48);
//
//
//	convolutional_layer<> c3(48, 56, 3, 48, 192, padding::same, true, 1, 1);
//	slice_layer sl3(shape3d(48, 56, 192), slice_type::slice_channels, 2);
//	linear_layer<> l31(56 * 48 * 96);
//	linear_layer<> l32(56 * 48 * 96);
//	elementwise_max_layer max_elt3(2, 56 * 48 * 96);
//	max_pooling_layer<> pl3(48, 56, 96, 2);
//
//	convolutional_layer<> c4(24, 28, 1, 96, 192, padding::same, true, 1, 1);
//	slice_layer sl4(shape3d(24, 28, 192), slice_type::slice_channels, 2);
//	linear_layer<> l41(28 * 24 * 96);
//	linear_layer<> l42(28 * 24 * 96);
//	elementwise_max_layer max_elt4(2, 28 * 24 * 96);
//
//
//	convolutional_layer<> c5(24, 28, 3, 96, 384, padding::same, true, 1, 1);
//	slice_layer sl5(shape3d(24, 28, 384), slice_type::slice_channels, 2);
//	linear_layer<> l51(28 * 24 * 192);
//	linear_layer<> l52(28 * 24 * 192);
//	elementwise_max_layer max_elt5(2, 28 * 24 * 192);
//	max_pooling_layer<> pl5(24, 28, 192, 2);
//
//	convolutional_layer<> c6(12, 14, 1, 192, 384, padding::same, true, 1, 1);
//	slice_layer sl6(shape3d(12, 14, 384), slice_type::slice_channels, 2);
//	linear_layer<> l61(14 * 12 * 192);
//	linear_layer<> l62(14 * 12 * 192);
//	elementwise_max_layer max_elt6(2, 14 * 12 * 192);
//
//
//	convolutional_layer<> c7(12, 14, 3, 192, 256, padding::same, true, 1, 1);
//	slice_layer sl7(shape3d(12, 14, 256), slice_type::slice_channels, 2);
//	linear_layer<> l71(14 * 12 * 128);
//	linear_layer<> l72(14 * 12 * 128);
//	elementwise_max_layer max_elt7(2, 14 * 12 * 128);
//
//
//	convolutional_layer<> c8(12, 14, 1, 128, 256, padding::same, true, 1, 1);
//	slice_layer sl8(shape3d(12, 14, 256), slice_type::slice_channels, 2);
//	linear_layer<> l81(14 * 12 * 128);
//	linear_layer<> l82(14 * 12 * 128);
//	elementwise_max_layer max_elt8(2, 14 * 12 * 128);
//
//
//	convolutional_layer<> c9(12, 14, 3, 128, 256, padding::same, true, 1, 1);
//	slice_layer sl9(shape3d(12, 14, 256), slice_type::slice_channels, 2);
//	linear_layer<> l91(14 * 12 * 128);
//	linear_layer<> l92(14 * 12 * 128);
//	elementwise_max_layer max_elt9(2, 14 * 12 * 128);
//	max_pooling_layer<> pl9(12, 14, 128, 2);
//
//	fully_connected_layer<> fc10(7 * 6 * 128, 512);
//	slice_layer sl10(shape3d(1, 1, 512), slice_type::slice_channels, 2);
//	linear_layer<> l101(1 * 1 * 256);
//	linear_layer<> l102(1 * 1 * 256);
//	elementwise_max_layer max_elt10(2, 1 * 1 * 256);
//
//#ifdef USE_OPENCL
//	my_gpu_device.registerOp(c1);
//	my_gpu_device.registerOp(c2);
//#endif
//	/*my_gpu_device.registerOp(c3);
//	my_gpu_device.registerOp(c4);
//	my_gpu_device.registerOp(c5);
//	my_gpu_device.registerOp(c6);
//	my_gpu_device.registerOp(c7);
//	my_gpu_device.registerOp(c8);
//	my_gpu_device.registerOp(c9);*/
//	cout << "num prog  " << ProgramManager::getInstance().num_programs() << endl;
//	//my_gpu_device.registerOp(c2);
//	cout << "num prog  " << ProgramManager::getInstance().num_programs() << endl;
//	network<graph>* nn = new network<graph>();
//	NN = (cv_handle_t)(nn);
//	// sl1 << (l11 , l12) ;
//	c1 << sl1 << (l11, l12) << max_elt1;
//	cout << "construct 1" << endl;
//	max_elt1 << pl1 << c2 << sl2 << (l21, l22) << max_elt2;
//	cout << "construct 2" << endl;
//	max_elt2 << c3 << sl3 << (l31, l32) << max_elt3;
//	cout << "construct 3" << endl;
//	max_elt3 << pl3 << c4 << sl4 << (l41, l42) << max_elt4;
//	cout << "construct 4" << endl;
//	max_elt4 << c5 << sl5 << (l51, l52) << max_elt5;
//	cout << "construct 5" << endl;
//	max_elt5 << pl5 << c6 << sl6 << (l61, l62) << max_elt6;
//	cout << "construct 6" << endl;
//	max_elt6 << c7 << sl7 << (l71, l72) << max_elt7;
//	cout << "construct 7" << endl;
//	max_elt7 << c8 << sl8 << (l81, l82) << max_elt8;
//	cout << "construct 8" << endl;
//	max_elt8 << c9 << sl9 << (l91, l92) << max_elt9;
//	cout << "construct 9" << endl;
//	max_elt9 << pl9 << fc10 << sl10 << (l101, l102) << max_elt10;
//	construct_graph(*nn, { &c1 }, { &max_elt10 });
//	cout << "construct finish" << endl;
//	cout << " nn->in_data_size() " << nn->in_data_size() << endl;
//	cout << nn << " == " << NN << endl;
//	load_weights_conv_fromfile(".\\weight\\conv1.dat", &c1);
//	load_weights_conv_fromfile(".\\weight\\conv2a.dat", &c2);
//	load_weights_conv_fromfile(".\\weight\\conv2.dat", &c3);
//	load_weights_conv_fromfile(".\\weight\\conv3a.dat", &c4);
//	load_weights_conv_fromfile(".\\weight\\conv3.dat", &c5);
//	load_weights_conv_fromfile(".\\weight\\conv4a.dat", &c6);
//	load_weights_conv_fromfile(".\\weight\\conv4.dat", &c7);
//	load_weights_conv_fromfile(".\\weight\\conv5a.dat", &c8);
//	load_weights_conv_fromfile(".\\weight\\conv5.dat", &c9);
//	load_weights_fullyconnected_fromfile(".\\weight\\fc1.dat", &fc10);
//	return 1;
//}

DNN_recogniton::DNN_recogniton( string pca_mean, string pca_proj)
{
	init_net();
	
	
	//load pca parameter
	mean_feat = new float[FEAT_NUM];
	cout << pca_mean.c_str() << endl;
	FILE* f_mean = fopen(pca_mean.c_str(), "rb");
	if (f_mean == NULL)
		cout << "open failed" << endl;
	//fread(feat_mat_.data, sizeof(float), FEAT_NUM, f_mean);
	//cout << feat_mat_.size() << endl;
	
	for (int i = 0; i < FEAT_NUM; i++)
	{
		fread(mean_feat+i, sizeof(float), 1, f_mean);
		//cout << feat_mat_.at<float>(0, i) << endl;
	}
	fclose(f_mean);
	//cout << feat_mat_ << endl;
	
	FILE* f_proj = fopen(pca_proj.c_str(), "rb");
	cout << pca_proj.c_str() << endl;
	if (f_proj == NULL)
		cout << "open failed" << endl;
	//fread(PCA_Proj.data, sizeof(float), FEAT_NUM*FEAT_PCA_NUM, f_proj);
		PCA_Proj = new float*[FEAT_PCA_NUM];//col index for fast product
	
	for (int i = 0; i < FEAT_PCA_NUM; i++)
	{
		PCA_Proj[i] = new float[FEAT_NUM];		
	}
	for (int i = 0; i < FEAT_NUM; i++)
	{
		for (int j = 0; j < FEAT_PCA_NUM; j++)
		{
			fread((void *)(PCA_Proj[j] + i), sizeof(float), 1, f_proj);
			//cout <<" pca_proj : "<< *(PCA_Proj[j] + i) << endl;
			//PCA_Proj->at<float>(i, j) = *m;
			//cout << i << "  " << j <<" "<<m<<endl;
			//cout << PCA_Proj->at<float>(i, j) << endl;
		}
	}

	
	//cout << PCA_Proj->at<float>(255, 99) << endl;
	//delete m;
	fclose(f_proj);
	feat_mat_ = new float[FEAT_NUM];
}

int DNN_recogniton::DNN_getfeat(ImageData aligned_face, float* feat)
{
	//load feat by layer named feat_layer_name
	DNN_forward(aligned_face, feat_mat_);
	/*cout << endl << "feat_mat_: ";
	for (int i = 0; i < 10; i++)
	{
		cout << feat_mat_[i] << "  ";
	}
	cout << endl;*/
	for (int i = 0; i < FEAT_NUM; i++)
	{
		feat_mat_[i] -= mean_feat[i];
	}
	for (int i = 0; i < FEAT_PCA_NUM; i++)
	{
		float sum = 0;
		float* pca_col = PCA_Proj[i];
		for (int j = 0; j < FEAT_NUM; j++)
		{
			sum += feat_mat_[j] * pca_col[j];
		}
		feat[i] = sum;
	}
	/*cout << endl << "feat: ";
	for (int i = 0; i < 10; i++)
	{
		cout << feat[i] << "  ";
	}
	cout << endl;*/
	//cout << "feat :";
	/*ofstream off("feat_.txt", ios::out);
	for (int i = 0; i < FEAT_NUM; i++)
		off << feat_mat_[i] << "  ";
	off << endl<<endl<<endl;
	off << *mean_feat << endl << endl;
	off << *PCA_Proj << endl<<endl;
	
	project the feat from feat_layer to pca space
	/*Mat feat_pca_mat = ((*feat_mat_) - (*mean_feat))*(*PCA_Proj);
	off << feat_pca_mat << endl;
	off.close();
	memcpy((void*)feat, feat_pca_mat.data, FEAT_PCA_NUM * sizeof(float));*/
	return 1;
}

int DNN_recogniton::DNN_forward(ImageData aligned_face, float* feat_mat)
{
	vec_t in_data(112 * 96 * 3);

	// generate random variables
	//uniform_rand(in_data.begin(), in_data.end(), 0, 1);
	/*cout << endl << "aligned_face: ";
	for (int i = 0; i < 10; i++)
	{
		cout << (float)aligned_face.data[i] << "  ";
	}
	cout << endl;*/
	ptr2vec_t(in_data, aligned_face);
	cout << "NN "<<NN << endl;
	auto res = ((network<graph>*)NN)->predict(in_data);
	/*cout << "in_data: ";
	for (int i = 0; i < 10; i++)
	{
		cout << in_data[i] << "  ";
	}
	cout <<endl<< "res: ";
	for (int i = 0; i < 10; i++)
	{
		cout << res[i] << "  ";
	}
	cout << endl;
	memcpy(feat_mat, &res,256 * sizeof(float));*/
	for (int i = 0; i < 256; i++)
	{
		feat_mat[i] = res[i];

	}
	return 1;
}


DNN_recogniton::~DNN_recogniton()
{
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
//void DNN_recogniton::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//	Blob<float>* input_layer = net_->input_blobs()[0];
//
//	int width = input_layer->width();
//	int height = input_layer->height();
//	float* input_data = input_layer->mutable_cpu_data();
//	for (int i = 0; i < input_layer->channels(); ++i) {
//		cv::Mat channel(height, width, CV_32FC1, input_data);
//		input_channels->push_back(channel);
//		input_data += width * height;
//	}
//}
//
//
//void DNN_recogniton::Preprocess(const cv::Mat& img,
//	std::vector<cv::Mat>* input_channels) {
//	/* Convert the input image to the input image format of the network. */
//	cv::Mat sample = img;
//
//	cv::Mat sample_resized;
//	if (sample.size() != input_geometry_)
//		cv::resize(sample, sample_resized, input_geometry_);
//	else
//		sample_resized = sample;
//
//	cv::Mat sample_float;
//	if (num_channels_ == 3)
//		sample_resized.convertTo(sample_float, CV_32FC3);
//	else
//		sample_resized.convertTo(sample_float, CV_32FC1);
//
//	//normalize
//	ofstream ofa("ofa.txt",ios::out);
//	sample_float = (sample_float - 127.5) / 128;
//	ofa << sample_float << endl;
//
//	ofa.close();
//
//	/* This operation will write the separate BGR planes directly to the
//	* input layer of the network because it is wrapped by the cv::Mat
//	* objects in input_channels. */
//	cv::split(sample_float, *input_channels);
//
//	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//		== net_->input_blobs()[0]->cpu_data())
//		<< "Input channels are not wrapping the input layer of the network.";
//}
void ptr2vec_t(vec_t& in, const ImageData& im)
{
	int width = im.width;
	int height = im.height;
	int channel = im.num_channels;

	unsigned char* ptr_im = im.data;
	float** ptr_in = new float*[channel];
	for (int chn = 0; chn < channel; chn++)
	{
		ptr_in[chn] = &in[chn*width*height];
	}
	for (int r = 0; r < height; r++)
	{
		unsigned char* im_row = ptr_im + r*width*channel;
		for (int c = 0; c < width; c++)
		{
			unsigned char* im_row_col = im_row + c*channel;
			for (int chn = 0; chn < channel; chn++)
			{
				ptr_in[chn][r*width + c] = (float)im_row_col[chn];
				ptr_in[chn][r*width + c] = (ptr_in[chn][r*width + c] - 127.5) / 128;
			}
		}
	}

}
void cimage2vec_t(vec_t &in, CImage& im)
{
	int width = im.GetWidth();
	int height = im.GetHeight();
	int channel = im.GetBPP() / 8;
	int nstep = im.GetPitch();
	unsigned char* ptr_im = (unsigned char*)im.GetBits();
	float** ptr_in = new float*[channel];
	for (int chn = 0; chn < channel; chn++)
	{
		ptr_in[chn] = &in[chn*width*height];
	}
	for (int r = 0; r < height; r++)
	{
		unsigned char* im_row = ptr_im + r*nstep;
		for (int c = 0; c < width; c++)
		{
			unsigned char* im_row_col = im_row + c*channel;
			for (int chn = 0; chn < channel; chn++)
			{
				ptr_in[chn][r*width + c] = (float)im_row_col[chn];
				ptr_in[chn][r*width + c] = (ptr_in[chn][r*width + c] - 127.5) / 128;
				//cout << ptr_in[chn][r*width + c] << "  ";
			}
			//cout << endl;
		}
	}

}