/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "stdafx.h"
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include "Dnn_recognition.h"
//#include "..\..\..\tiny_dnn\models\alexnet.h"

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
#include "Dnn_recognition.h"
using namespace std;
#include "atlimage.h"
void cimage2vec_t(vec_t &in, CImage& im);
void cimage2imagedata(ImageData &img, const CImage &im);
#if defined(USE_OPENCL) || defined(USE_CUDA)
device_t device_type(size_t *platform, size_t *device) {
	// check which platforms are available
	auto platforms = CLCudaAPI::GetAllPlatforms();

	// if no platforms - return -1
	if (platforms.size() == 0) {
		return device_t::NONE;
	}
	
	std::array<std::string, 2> devices_order = { "GPU", "CPU" };
	std::map<std::string, device_t>
		devices_t_order = { std::make_pair("GPU", device_t::GPU),
		std::make_pair("CPU", device_t::CPU) };
	for (auto d_type : devices_order)
	for (auto p = platforms.begin(); p != platforms.end(); ++p)
	for (size_t d = 0; d < p->NumDevices(); ++d) {
		auto dev = CLCudaAPI::Device(*p, d);
		if (dev.Type() == d_type) {
			*platform = p - platforms.begin();
			*device = d;
			cout <<(int) d  << endl;
			return devices_t_order[d_type];
		}
	}
	// no CPUs or GPUs
	return device_t::NONE;
}

#define TINY_DNN_GET_DEVICE_AND_PLATFORM       \
	size_t cl_platform = 0, cl_device = 0; \
	device_t device = device_type(&cl_platform, &cl_device);
#else
#define TINY_DNN_GET_DEVICE_AND_PLATFORM       \
	size_t cl_platform = 0, cl_device = 0; \
	device_t device = device_t::NONE;
#endif  // defined(USE_OPENCL) || defined(USE_CUDA)


void testhaha() {
	network<graph> net1, net2;
	vec_t in = { 1, 2, 3 };

	fully_connected_layer<tan_h> f1(3, 4);
	slice_layer s1(shape3d(2, 1, 2), slice_type::slice_channels, 2);
	fully_connected_layer<softmax> f2(2, 2);
	fully_connected_layer<elu> f3(2, 2);
	elementwise_add_layer c4(2, 2);

	f1 << s1;
	s1 << (f2, f3) << c4;

	construct_graph(net1, { &f1 }, { &c4 });

	
}





void sample1_convne2();
void lightcnn();
void test_sdk();
//void sample1_convnet(const string& data_dir = "../../../data");
//void sample2_mlp(const string& data_dir = "../../data");
//void sample3_dae();
//void sample4_dropout(const string& data_dir = "../../data");
//void sample5_unbalanced_training_data(const string& data_dir = "../../data");
void sample6_graph();

int main(int argc, char** argv) {
	try {
		if (argc == 2) {
			///sample1_convnet(argv[1]);
		}
		else {
			testhaha();
			lightcnn();
			//sample6_graph();
			//sample1_convne2();
			//sample1_convnet();
			//test_sdk();
		}
		
		
	}
	catch (const nn_error& e) {
		std::cout << e.what() << std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
//void sample1_convnet(const string& data_dir) {
//	
//	
//	// construct LeNet-5 architecture
//	network<sequential> nn;
//	adagrad optimizer;
//
//	// connection table [Y.Lecun, 1998 Table.1]
//#define O true
//#define X false
//	static const bool connection[] = {
//		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
//		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
//		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
//		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
//		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
//		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
//	};
//#undef O
//#undef X
//
//	nn << convolutional_layer<tan_h>(
//		32, 32, 5, 1, 6)  /* 32x32 in, 5x5 kernel, 1-6 fmaps conv */
//		<< average_pooling_layer<tan_h>(
//		28, 28, 6, 2)     /* 28x28 in, 6 fmaps, 2x2 subsampling */
//		<< convolutional_layer<tan_h>(
//		14, 14, 5, 6, 16, connection_table(connection, 6, 16))
//		<< average_pooling_layer<tan_h>(10, 10, 16, 2)
//		<< convolutional_layer<tan_h>(5, 5, 5, 16, 120)
//		<< fully_connected_layer<tan_h>(120, 10);
//
//	std::cout << "load models..." << std::endl;
//
//	// load MNIST dataset
//	std::vector<label_t> train_labels, test_labels;
//	std::vector<vec_t>   train_images, test_images;
//
//	std::string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
//	std::string train_images_path = data_dir + "/train-images.idx3-ubyte";
//	std::string test_labels_path = data_dir + "/t10k-labels.idx1-ubyte";
//	std::string test_images_path = data_dir + "/t10k-images.idx3-ubyte";
//
//	parse_mnist_labels(train_labels_path, &train_labels);
//	parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 2, 2);
//	parse_mnist_labels(test_labels_path, &test_labels);
//	parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 2, 2);
//
//	std::cout << "start learning" << std::endl;
//
//	progress_display disp(train_images.size());
//	timer t;
//	int minibatch_size = 10;
//
//	optimizer.alpha *= std::sqrt(minibatch_size);
//
//	// create callback
//	auto on_enumerate_epoch = [&](){
//		std::cout << t.elapsed() << "s elapsed." << std::endl;
//
//		tiny_dnn::result res = nn.test(test_images, test_labels);
//
//		std::cout << res.num_success << "/" << res.num_total << std::endl;
//
//		disp.restart(train_images.size());
//		t.restart();
//	};
//
//	auto on_enumerate_minibatch = [&](){
//		disp += minibatch_size;
//	};
//
//	// training
//	nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, 20,
//		on_enumerate_minibatch, on_enumerate_epoch);
//
//	std::cout << "end training." << std::endl;
//
//	// test and show results
//	nn.test(test_images, test_labels).print_detail(std::cout);
//
//	// save networks
//	std::ofstream ofs("LeNet-weights");
//	ofs << nn;
//}
void sample1_convne2() {


	// construct LeNet-5 architecture
	network<sequential> nn;
/*
	nn << conv<relu>(227, 227, 11, 11, 3, 96, padding::valid, true, 4, 4)
		<< lrn<activation::identity>(55, 55, 5, 96,0.0001,0.75)
		<< max_pool<activation::identity>(55, 55, 96, 2)
		<< conv<relu>(27, 27, 5, 5, 96, 256, padding::valid, true, 1, 1)
		<< lrn<activation::identity>(23, 23, 5, 256, 0.0001, 0.75)
		<< max_pool<activation::identity>(23, 23, 256, 1)
	    << conv<relu>(23, 23, 3, 3, 256, 384, padding::valid, true, 1, 1)
		 << conv<relu>(21, 21, 3, 3, 384, 384, padding::valid, true, 1, 1)
		<< conv<relu>(19, 19, 3, 3, 384, 256, padding::valid, true, 1, 1)
		<< max_pool<activation::identity>(17, 17, 256, 2)
		<< fc<relu>(16384, 4096)
		 << dropout(4096, 0.5)
		 << fc<relu>(4096, 4096)
		 << dropout(4096, 0.5);*/
	nn.load("model.dat");
	cout << "load model.dat success" << endl;
	vec_t in(227* 227 * 3);

	CImage im;
	LPTSTR im_name = L"test227.jpg";
	im.Load(im_name);
	cout << "width: " << im.GetWidth() << "height: " << im.GetHeight() << "channel: " << im.GetBPP() / 8 << endl;
	

	// generate random variables
	//uniform_rand(in_data.begin(), in_data.end(), 0, 1);
	cimage2vec_t(in, im);

	double t=clock(); // start the timer
	//for (int i = 0; i < 10;i++)
		auto res = nn.predict(in);
		cout << "res : ";
		for (int i = 0; i < 10; i++)
			cout << res[i] << "  ";
		cout << endl;
	double elapsed_ms = clock()-t;
	

	cout << "Elapsed time(ms): " << elapsed_ms << endl;
	

}


///////////////////////////////////////////////////////////////////////////////
// learning 3-Layer Networks
//void sample2_mlp(const string& data_dir) {
//	const cnn_size_t num_hidden_units = 500;
//
//#if defined(_MSC_VER) && _MSC_VER < 1800
//	 initializer-list is not supported
//	int num_units[] = { 28 * 28, num_hidden_units, 10 };
//	auto nn = make_mlp<tan_h>(num_units, num_units + 3);
//#else
//	auto nn = make_mlp<tan_h>({ 28 * 28, num_hidden_units, 10 });
//#endif
//	gradient_descent optimizer;
//
//	 load MNIST dataset
//	std::vector<label_t> train_labels, test_labels;
//	std::vector<vec_t>   train_images, test_images;
//
//	std::string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
//	std::string train_images_path = data_dir + "/train-images.idx3-ubyte";
//	std::string test_labels_path = data_dir + "/t10k-labels.idx1-ubyte";
//	std::string test_images_path = data_dir + "/t10k-images.idx3-ubyte";
//
//	parse_mnist_labels(train_labels_path, &train_labels);
//	parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 0, 0);
//	parse_mnist_labels(test_labels_path, &test_labels);
//	parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 0, 0);
//
//	optimizer.alpha = 0.001;
//
//	progress_display disp(train_images.size());
//	timer t;
//
//	 create callback
//	auto on_enumerate_epoch = [&](){
//		std::cout << t.elapsed() << "s elapsed." << std::endl;
//
//		tiny_dnn::result res = nn.test(test_images, test_labels);
//
//		std::cout << optimizer.alpha << ","
//			<< res.num_success << "/" << res.num_total << std::endl;
//
//		optimizer.alpha *= 0.85;  // decay learning rate
//		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);
//
//		disp.restart(train_images.size());
//		t.restart();
//	};
//
//	auto on_enumerate_data = [&](){
//		++disp;
//	};
//
//	nn.train<mse>(optimizer, train_images, train_labels, 1, 20,
//		on_enumerate_data, on_enumerate_epoch);
//}

///////////////////////////////////////////////////////////////////////////////
// denoising auto-encoder
//void sample3_dae() {
//#if defined(_MSC_VER) && _MSC_VER < 1800
//	// initializer-list is not supported
//	int num_units[] = { 100, 400, 100 };
//	auto nn = make_mlp<tan_h>(num_units, num_units + 3);
//#else
//	auto nn = make_mlp<tan_h>({ 100, 400, 100 });
//#endif
//
//	std::vector<vec_t> train_data_original;
//
//	// load train-data
//
//	std::vector<vec_t> train_data_corrupted(train_data_original);
//
//	for (auto& d : train_data_corrupted) {
//		d = corrupt(move(d), 0.1, 0.0);  // corrupt 10% data
//	}
//
//	gradient_descent optimizer;
//
//	// learning 100-400-100 denoising auto-encoder
//	nn.train<mse>(optimizer, train_data_corrupted, train_data_original);
//}

///////////////////////////////////////////////////////////////////////////////
// dropout-learning

//void sample4_dropout(const string& data_dir) {
//	typedef network<sequential> Network;
//	Network nn;
//	cnn_size_t input_dim = 28 * 28;
//	cnn_size_t hidden_units = 800;
//	cnn_size_t output_dim = 10;
//	gradient_descent optimizer;
//
//	fully_connected_layer<tan_h> f1(input_dim, hidden_units);
//	dropout_layer dropout(hidden_units, 0.5);
//	fully_connected_layer<tan_h> f2(hidden_units, output_dim);
//	nn << f1 << dropout << f2;
//
//	optimizer.alpha = 0.003;  // TODO(nyanp): not optimized
//	optimizer.lambda = 0.0;
//
//	// load MNIST dataset
//	std::vector<label_t> train_labels, test_labels;
//	std::vector<vec_t>   train_images, test_images;
//
//	std::string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
//	std::string train_images_path = data_dir + "/train-images.idx3-ubyte";
//	std::string test_labels_path = data_dir + "/t10k-labels.idx1-ubyte";
//	std::string test_images_path = data_dir + "/t10k-images.idx3-ubyte";
//
//	parse_mnist_labels(train_labels_path, &train_labels);
//	parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 0, 0);
//	parse_mnist_labels(test_labels_path, &test_labels);
//	parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 0, 0);
//
//	// load train-data, label_data
//	progress_display disp(train_images.size());
//	timer t;
//
//	// create callback
//	auto on_enumerate_epoch = [&](){
//		std::cout << t.elapsed() << "s elapsed." << std::endl;
//
//		dropout.set_context(net_phase::test);
//		tiny_dnn::result res = nn.test(test_images, test_labels);
//		dropout.set_context(net_phase::train);
//
//		std::cout << optimizer.alpha << ","
//			<< res.num_success << "/" << res.num_total << std::endl;
//
//		optimizer.alpha *= 0.99;  // decay learning rate
//		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);
//
//		disp.restart(train_images.size());
//		t.restart();
//	};
//
//	auto on_enumerate_data = [&](){
//		++disp;
//	};
//
//	nn.train<mse>(optimizer, train_images, train_labels, 1, 100,
//		on_enumerate_data, on_enumerate_epoch);
//
//	// change context to enable all hidden-units
//	// f1.set_context(dropout::test_phase);
//	// std::cout << res.num_success << "/" << res.num_total << std::endl;
//}

#include "tiny_dnn/util/target_cost.h"

///////////////////////////////////////////////////////////////////////////////
// learning unbalanced training data

//void sample5_unbalanced_training_data(const string& data_dir) {
//	// keep the network relatively simple
//	
//	const cnn_size_t num_hidden_units = 20;
//	auto nn_balanced = make_mlp<tan_h>({ 28 * 28, num_hidden_units, 10 });
//	gradient_descent optimizer;
//
//	// load MNIST dataset
//	std::vector<label_t> train_labels, test_labels;
//	std::vector<vec_t>   train_images, test_images;
//
//	std::string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
//	std::string train_images_path = data_dir + "/train-images.idx3-ubyte";
//	std::string test_labels_path = data_dir + "/t10k-labels.idx1-ubyte";
//	std::string test_images_path = data_dir + "/t10k-images.idx3-ubyte";
//
//	parse_mnist_labels(train_labels_path, &train_labels);
//	parse_mnist_images(train_images_path, &train_images, -1.0, 1.0, 0, 0);
//	parse_mnist_labels(test_labels_path, &test_labels);
//	parse_mnist_images(test_images_path, &test_images, -1.0, 1.0, 0, 0);
//
//	{  // create an unbalanced training set
//		std::vector<label_t> train_labels_unbalanced;
//		std::vector<vec_t>   train_images_unbalanced;
//		train_labels_unbalanced.reserve(train_labels.size());
//		train_images_unbalanced.reserve(train_images.size());
//
//		for (size_t i = 0, end = train_labels.size(); i < end; ++i) {
//			const label_t label = train_labels[i];
//
//			// drop most 0s, 1s, 2s, 3s, and 4s
//			const bool keep_sample = label >= 5 || bernoulli(0.005);
//
//			if (keep_sample) {
//				train_labels_unbalanced.push_back(label);
//				train_images_unbalanced.push_back(train_images[i]);
//			}
//		}
//
//		// keep the newly created unbalanced training set
//		std::swap(train_labels, train_labels_unbalanced);
//		std::swap(train_images, train_images_unbalanced);
//	}
//
//	optimizer.alpha = 0.001;
//
//	progress_display disp(train_images.size());
//	timer t;
//
//	const int minibatch_size = 10;
//
//	auto nn = &nn_standard;  // the network referred to by the callbacks
//
//	// create callbacks - as usual
//	auto on_enumerate_epoch = [&](){
//		std::cout << t.elapsed() << "s elapsed." << std::endl;
//
//		tiny_dnn::result res = nn->test(test_images, test_labels);
//
//		std::cout << optimizer.alpha << ","
//			<< res.num_success << "/" << res.num_total << std::endl;
//
//		optimizer.alpha *= 0.85;  // decay learning rate
//		optimizer.alpha = std::max(
//			static_cast<tiny_dnn::float_t>(0.00001), optimizer.alpha);
//
//		disp.restart(train_images.size());
//		t.restart();
//	};
//
//	auto on_enumerate_data = [&](){
//		disp += minibatch_size;
//	};
//
//	// first train the standard network (default cost - equal for each sample)
//	// - note that it does not learn the classes 0-4
//	nn_standard.train<mse>(optimizer, train_images, train_labels,
//		minibatch_size, 20, on_enumerate_data,
//		on_enumerate_epoch, true, CNN_TASK_SIZE);
//
//	// then train another network, now with explicitly
//	// supplied target costs (aim: a more balanced predictor)
//	// - note that it can learn the classes 0-4 (at least somehow)
//	nn = &nn_balanced;
//	optimizer = gradient_descent();
//	const auto target_cost = create_balanced_target_cost(train_labels, 0.8);
//	nn_balanced.train<mse>(optimizer, train_images, train_labels,
//		minibatch_size, 20, on_enumerate_data,
//		on_enumerate_epoch, true, CNN_TASK_SIZE,
//		target_cost);
//
//	// test and show results
//	std::cout << "\nStandard training (implicitly assumed equal "
//		<< "cost for every sample):\n";
//	nn_standard.test(test_images, test_labels).print_detail(std::cout);
//
//	std::cout << "\nBalanced training "
//		<< "(explicitly supplied target costs):\n";
//	nn_balanced.test(test_images, test_labels).print_detail(std::cout);
//}


void sample6_graph() {
	// declare node

	auto in1 = std::make_shared<input_layer>(shape3d(3, 1, 1));
	auto in2 = std::make_shared<input_layer>(shape3d(3, 1, 1));
	auto added = std::make_shared<max_elt>(2, 3);
	auto out = std::make_shared<linear_layer<relu>>(3);

	// connect
	(in1, in2) << added;
	added << out;

	network<graph> net;
	construct_graph(net, { in1, in2 }, { out });

	auto res = net.predict({ { 2, 4, 3 }, { -1, 2, 5 } })[0];

	// relu({2,4,3} + {-1,2,-5}) = {1,6,0}
	std::cout << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "," << res[4] << "," << res[5] << "," << res[6] << "," << res[7] << std::endl;
}

//typedef struct ImageData {
//	ImageData() {
//		data = nullptr;
//		width = 0;
//		height = 0;
//		num_channels = 0;
//	}
//
//	ImageData(int32_t img_width, int32_t img_height,
//		int32_t img_num_channels = 1) {
//		data = nullptr;
//		width = img_width;
//		height = img_height;
//		num_channels = img_num_channels;
//	}
//
//	uint8_t* data;
//	int32_t width;
//	int32_t height;
//	int32_t num_channels;
//} ImageData;

void lightcnn()
{
#if 1
	auto in = std::make_shared<input_layer>(shape3d(96, 112, 3));
	convolutional_layer<> c1(96, 112, 5, 3, 96, padding::same, true, 1, 1);
	cout << "c1: " << c1.in_channels() << "  " << c1.out_channels() << endl;
	slice_layer sl1(shape3d(96, 112, 96), slice_type::slice_channels, 2); //shape3d(112, 96, 96)
	linear_layer<> l11(112 * 96 * 48);
	linear_layer<> l12(112 * 96 * 48);
	elementwise_max_layer max_elt1(2, 112 * 96 * 48);
	max_pooling_layer<> pl1(96, 112, 48, 2);

	convolutional_layer<> c2(48, 56, 1, 48, 96, padding::same, true, 1, 1);
	slice_layer sl2(shape3d(48, 56, 96), slice_type::slice_channels, 2);
	linear_layer<> l21(56 * 48 * 48);
	linear_layer<> l22(56 * 48 * 48);
	elementwise_max_layer max_elt2(2, 56 * 48 * 48);


	convolutional_layer<> c3(48, 56, 3, 48, 192, padding::same, true, 1, 1);
	slice_layer sl3(shape3d(48, 56, 192), slice_type::slice_channels, 2);
	linear_layer<> l31(56 * 48 * 96);
	linear_layer<> l32(56 * 48 * 96);
	elementwise_max_layer max_elt3(2, 56 * 48 * 96);
	max_pooling_layer<> pl3(48, 56, 96, 2);

	convolutional_layer<> c4(24, 28, 1, 96, 192, padding::same, true, 1, 1);
	slice_layer sl4(shape3d(24, 28, 192), slice_type::slice_channels, 2);
	linear_layer<> l41(28 * 24 * 96);
	linear_layer<> l42(28 * 24 * 96);
	elementwise_max_layer max_elt4(2, 28 * 24 * 96);


	convolutional_layer<> c5(24, 28, 3, 96, 384, padding::same, true, 1, 1);
	slice_layer sl5(shape3d(24, 28, 384), slice_type::slice_channels, 2);
	linear_layer<> l51(28 * 24 * 192);
	linear_layer<> l52(28 * 24 * 192);
	elementwise_max_layer max_elt5(2, 28 * 24 * 192);
	max_pooling_layer<> pl5(24, 28, 192, 2);

	convolutional_layer<> c6(12, 14, 1, 192, 384, padding::same, true, 1, 1);
	slice_layer sl6(shape3d(12, 14, 384), slice_type::slice_channels, 2);
	linear_layer<> l61(14 * 12 * 192);
	linear_layer<> l62(14 * 12 * 192);
	elementwise_max_layer max_elt6(2, 14 * 12 * 192);


	convolutional_layer<> c7(12, 14, 3, 192, 256, padding::same, true, 1, 1);
	slice_layer sl7(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<> l71(14 * 12 * 128);
	linear_layer<> l72(14 * 12 * 128);
	elementwise_max_layer max_elt7(2, 14 * 12 * 128);


	convolutional_layer<> c8(12, 14, 1, 128, 256, padding::same, true, 1, 1);
	slice_layer sl8(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<> l81(14 * 12 * 128);
	linear_layer<> l82(14 * 12 * 128);
	elementwise_max_layer max_elt8(2, 14 * 12 * 128);


	convolutional_layer<> c9(12, 14, 3, 128, 256, padding::same, true, 1, 1);
	slice_layer sl9(shape3d(12, 14, 256), slice_type::slice_channels, 2);
	linear_layer<> l91(14 * 12 * 128);
	linear_layer<> l92(14 * 12 * 128);
	elementwise_max_layer max_elt9(2, 14 * 12 * 128);
	max_pooling_layer<> pl9(12, 14, 128, 2);

	fully_connected_layer<> fc10(7 * 6 * 128, 512);
	slice_layer sl10(shape3d(1, 1, 512), slice_type::slice_channels, 2);
	linear_layer<> l101(1 * 1 * 256);
	linear_layer<> l102(1 * 1 * 256);
	elementwise_max_layer max_elt10(2, 1 * 1 * 256);

#else

	auto in = std::make_shared<input_layer>(shape3d(3, 112, 96));
	convolutional_layer<> c1(112, 96, 5, 3, 96, padding::same, true, 1, 1);// , core::backend_t::opencl);
	cout << "c1: " << c1.in_channels() << "  " << c1.out_channels() << endl;
	slice_layer sl1(shape3d(112, 96, 96), slice_type::slice_channels, 2); //shape3d(112, 96, 96)
	linear_layer<> l11(112*96*48);
	linear_layer<> l12(112*96*48);
	elementwise_max_layer max_elt1(2, 112*96*48);
	max_pooling_layer<> pl1(112, 96, 48, 2);

	convolutional_layer<> c2(56, 48, 1, 48, 96, padding::same, true, 1, 1);// , core::backend_t::opencl);
	slice_layer sl2(shape3d(56, 48, 96), slice_type::slice_channels, 2);
	linear_layer<> l21(56*48*48);
	linear_layer<> l22(56*48*48);
	elementwise_max_layer max_elt2(2, 56*48*48);
	

	convolutional_layer<> c3(56, 48, 3, 48, 192, padding::same, true, 1, 1);
	slice_layer sl3(shape3d(56, 48, 192), slice_type::slice_channels, 2);
	linear_layer<> l31(56*48*96);
	linear_layer<> l32(56*48*96);
	elementwise_max_layer max_elt3(2, 56*48*96);
	max_pooling_layer<> pl3(56, 48, 96, 2);

	convolutional_layer<> c4(28, 24, 1, 96, 192, padding::same, true, 1, 1);
	slice_layer sl4(shape3d(28, 24, 192), slice_type::slice_channels, 2);
	linear_layer<> l41(28*24*96);
	linear_layer<> l42(28*24*96);
	elementwise_max_layer max_elt4(2, 28*24*96);
	

	convolutional_layer<> c5(28, 24, 3, 96, 384, padding::same, true, 1, 1);
	slice_layer sl5(shape3d(28, 24, 384), slice_type::slice_channels, 2);
	linear_layer<> l51(28*24*192);
	linear_layer<> l52(28*24*192);
	elementwise_max_layer max_elt5(2, 28*24*192);
	max_pooling_layer<> pl5(28, 24, 192, 2);

	convolutional_layer<> c6(14, 12, 1, 192, 384, padding::same, true, 1, 1);
	slice_layer sl6(shape3d(14, 12, 384), slice_type::slice_channels, 2);
	linear_layer<> l61(14*12*192);
	linear_layer<> l62(14*12*192);
	elementwise_max_layer max_elt6(2, 14*12*192);
	

	convolutional_layer<> c7(14, 12, 3, 192, 256, padding::same, true, 1, 1);
	slice_layer sl7(shape3d(14, 12, 256), slice_type::slice_channels, 2);
	linear_layer<> l71(14*12*128);
	linear_layer<> l72(14*12*128);
	elementwise_max_layer max_elt7(2, 14*12*128);
	

	convolutional_layer<> c8(14, 12, 1, 128, 256, padding::same, true, 1, 1);
	slice_layer sl8(shape3d(14, 12, 256), slice_type::slice_channels, 2);
	linear_layer<> l81(14*12*128);
	linear_layer<> l82(14*12*128);
	elementwise_max_layer max_elt8(2, 14*12*128);


	convolutional_layer<> c9(14, 12, 3, 128, 256, padding::same, true, 1, 1);
	slice_layer sl9(shape3d(14, 12, 256), slice_type::slice_channels, 2);
	linear_layer<> l91(14*12*128);
	linear_layer<> l92(14*12*128);
	elementwise_max_layer max_elt9(2, 14*12*128);
	max_pooling_layer<> pl9(14, 12,128, 2);

	fully_connected_layer<> fc10(7*6*128, 512);
	slice_layer sl10(shape3d(1, 1, 512), slice_type::slice_channels, 2);
	linear_layer<> l101(1*1*256);
	linear_layer<> l102(1*1*256);
	elementwise_max_layer max_elt10(2, 1*1*256);
#endif

	network<graph> nn;
	// sl1 << (l11 , l12) ;
	c1 << sl1 << (l11, l12) << max_elt1;
	cout << "construct 1" << endl;
		max_elt1 << pl1 << c2 << sl2 << (l21, l22) << max_elt2;
		cout << "construct 2" << endl;
		max_elt2 << c3 << sl3 << (l31, l32) << max_elt3;
		cout << "construct 3" << endl;
		max_elt3 << pl3 << c4 << sl4 << (l41, l42) << max_elt4;
		cout << "construct 4" << endl;
		max_elt4  << c5 << sl5 << (l51, l52) << max_elt5;
		cout << "construct 5" << endl;
		max_elt5 << pl5 << c6 << sl6 << (l61, l62) << max_elt6;
		cout << "construct 6" << endl;
		max_elt6  << c7 << sl7 << (l71, l72) << max_elt7;
		cout << "construct 7" << endl;
		max_elt7  << c8 << sl8 << (l81, l82) << max_elt8;
		cout << "construct 8" << endl;
		max_elt8  << c9 << sl9 << (l91, l92) << max_elt9;
		cout << "construct 9" << endl;
		max_elt9 << pl9 << fc10 << sl10 << (l101, l102) << max_elt10;
	construct_graph(nn, { &c1 }, { &max_elt10 });
	cout << "construct finish" << endl;
	
	load_weights_conv_fromfile(".\\weight\\conv1.dat", &c1);
	load_weights_conv_fromfile(".\\weight\\conv2a.dat", &c2);
	load_weights_conv_fromfile(".\\weight\\conv2.dat", &c3);
	load_weights_conv_fromfile(".\\weight\\conv3a.dat", &c4);
	load_weights_conv_fromfile(".\\weight\\conv3.dat", &c5);
	load_weights_conv_fromfile(".\\weight\\conv4a.dat", &c6);
	load_weights_conv_fromfile(".\\weight\\conv4.dat", &c7);
	load_weights_conv_fromfile(".\\weight\\conv5a.dat", &c8);
	load_weights_conv_fromfile(".\\weight\\conv5.dat", &c9);
	load_weights_fullyconnected_fromfile(".\\weight\\fc1.dat", &fc10);
	
	CImage im;
	LPTSTR im_name = L"test.jpg";
	im.Load(im_name);
	cout << "width: " << im.GetWidth() << "height: " << im.GetHeight() << "channel: " << im.GetBPP()/8 << endl;
	vec_t in_data(112 * 96*3 );

	// generate random variables
	//uniform_rand(in_data.begin(), in_data.end(), 0, 1);
	cimage2vec_t(in_data, im);
	/*cout.precision(7);
	for (int i = 112*96+0; i < 112*96*2; i=i++)
		cout << in_data[i] << "  ";
	cout << endl;*/
	double t = clock(); // start the timer
	
	//for (int ii = 0; ii < 100;ii++)
		auto res = nn.predict(in_data);
		double ed = clock();
		cout << "c1: " << c1.in_channels() << "  " << c1.out_data_shape() << endl;
		cout << "c2: " << c2.in_channels() << "  " << c2.out_data_shape() << endl;
		cout << "c3: " << c3.in_channels() << "  " << c3.out_data_shape() << endl;
		cout << "c4: " << c4.in_channels() << "  " << c4.out_data_shape() << endl;
		cout << "c5: " << c5.in_channels() << "  " << c5.out_data_shape() << endl;
		shape3d num_out = c1.out_data_shape()[0];
		//vector<vec_t*> a = c1.weights();
		//vec_t a_0 = *(a[1]);
		vector<tensor_t> a1 = c1.output();
		vec_t a1_0 = (a1[0])[0];
		
		//for (size_t i = 5 * 5 * 96 * 3 - 10; i < 5 * 5 * 96 * 3 +10; i++)
		for (size_t i = 0; i < 20; i++)
		{
			cout << a1_0[i] << "  ";
		}
		cout << endl;
		vector<vec_t*> a2 = c1.weights();
		vec_t a2_0 = (a2[1])[0];

		//for (size_t i = 5 * 5 * 96 * 3 - 10; i < 5 * 5 * 96 * 3 +10; i++)
		for (size_t i = 0; i < 96; i++)
		{
			cout << a2_0[i] << "  ";
		}
		cout << endl;


		vector<vec_t*> a3 = c1.weights();
		vec_t a3_0 = (a3[0])[0];
		//for (size_t i = 5 * 5 * 96 * 3 - 10; i < 5 * 5 * 96 * 3 +10; i++)
		for (size_t i = 0; i < 25*3; i++)
		{
			cout << a3_0[i] << "  ";
		}
		cout << endl;
	cout << "res : ";
	for (int i = 0; i < 20; i++)
		cout << res[i] << "  ";
	cout << endl;
	double elapsed_ms = ed - t;

	cout << endl << "elapse " << elapsed_ms << endl<<endl;
	


}

void test_sdk()
{
	DNN_recogniton a_sdk = DNN_recogniton();
	float *feat = new float[100];
	CImage im;
	LPTSTR im_name = L"test.jpg";
	im.Load(im_name);
	ImageData img;
	cimage2imagedata(img, im);
	img.width = im.GetWidth();
	img.height = im.GetHeight();
	img.num_channels = im.GetBPP() / 8;
	cout << "width: " << img.width << "  height: " << img.height << "  channels: " << img.num_channels << endl;
	double st = clock();
	a_sdk.DNN_getfeat(img, feat);
	double ed = clock();
	cout << "feat: ";
	for (int i = 0; i < 10;i++)
	{
		cout << feat[i] << "  ";
	}
	cout << endl;
	cout << "elapse: " << ed - st << endl;
}
//void cimage2ptr(unsigned char * ptr, CImage &im)//opencv
//{
//	int width = im.GetWidth();
//	int height = im.GetHeight();
//	int channel = im.GetBPP() / 8;
//	int nstep = im.GetPitch();
//	unsigned char* ptr_cimage = (unsigned char*)im.GetBits();
//	for (int r = 0; r < height; r++)
//	{
//		unsigned char* ptr_row = ptr+r*width*channel;
//		unsigned char* im_row = ptr_cimage + r*nstep;
//		for (int c = 0; c < width; c++)
//		{
//			for (int chn = 0; chn < channel; chn++)
//			{
//				ptr_row[c*channel + chn] = ptr_cimage[c*channel + chn];
//			}
//		}
//	}
//
//}
////void ptr2vec_t(vec_t in, ImageData& im)
////{
////	int width = im.width;
////	int height = im.height;
////	int channel = im.num_channels;
////	
////	unsigned char* ptr_im = im.data;
////	float** ptr_in = new float*[channel];
////	for (int chn = 0; chn < channel; chn++)
////	{
////		ptr_in[chn] = &in[chn*width*height];
////	}
////	for (int r = 0; r < height; r++)
////	{
////		unsigned char* im_row = ptr_im + r*width*channel;
////		for (int c = 0; c < width; c++)
////		{
////			unsigned char* im_row_col = im_row + c*channel;
////			for (int chn = 0; chn < channel; chn++)
////			{
////				ptr_in[chn][r*width + c] = (float)im_row_col[chn];
////			}
////		}
////	}
////
////}
//
//void cimage2vec_t(vec_t &in, CImage& im)
//{
//	int width = im.GetWidth();
//	int height = im.GetHeight();
//	int channel = im.GetBPP()/8;
//	int nstep = im.GetPitch();
//	unsigned char* ptr_im = (unsigned char*)im.GetBits();
//	float** ptr_in = new float*[channel];
//	for (int chn = 0; chn < channel; chn++)
//	{
//		ptr_in[chn] = &in[chn*width*height];
//	}
//	for (int r = 0; r < height; r++)
//	{
//		unsigned char* im_row = ptr_im + r*nstep;
//		for (int c = 0; c < width; c++)
//		{
//			unsigned char* im_row_col = im_row + c*channel;
//			for (int chn = 0; chn < channel; chn++)
//			{
//				ptr_in[chn][r*width + c] = (float)im_row_col[chn];
//				ptr_in[chn][r*width + c] = (ptr_in[chn][r*width + c] - 127.5) / 128;
//				//cout << ptr_in[chn][r*width + c] << "  ";
//			}
//			//cout << endl;
//		}
//	}
//
//}

void cimage2imagedata(ImageData &img, const CImage &im)
{
	int width = im.GetWidth();
	int height = im.GetHeight();
	int channel = im.GetBPP() / 8;
	int nstep = im.GetPitch();
	unsigned char* ptr_cimage = (unsigned char*)im.GetBits();
	img.data = new unsigned char[width*height*channel];
	unsigned char* ptr_imagedata = (unsigned char*)img.data;

	img.height = height;
	img.width = width;
	img.num_channels = channel;

	for (size_t r = 0; r < height; r++)
	{
		for (size_t c = 0; c < width; c++)
		{
			for (size_t chn = 0; chn < channel; chn++)
			{
				*(ptr_imagedata + r*width*channel + c*channel + chn) = *(ptr_cimage + r*nstep + c*channel + chn);
			}
		}
	}
	
}