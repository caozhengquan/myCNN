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
#pragma once
#include "tiny_dnn/util/util.h"
#include "tiny_dnn/layers/fully_connected_layer.h"

namespace tiny_dnn {

/**
* create multi-layer perceptron
*/
template<typename activation, typename Iter>
network<sequential> make_mlp(Iter first, Iter last) {
    typedef network<sequential> net_t;
    net_t n;

    Iter next = first + 1;
    for (; next != last; ++first, ++next)
        n << fully_connected_layer<activation>(*first, *next);
    return n;
}

/**
 * create multi-layer perceptron
 */
template<typename activation>
network<sequential> make_mlp(const std::vector<cnn_size_t>& units) {
    return make_mlp<activation>(units.begin(), units.end());
}

inline void load_weights_conv_fromfile(const std::string src, layer *dst) {
	FILE* if_src = fopen(src.c_str(), "rb");
	// fill weight
	//TODO: check if it works
	//int out_channels = dst->out_shape().depth_;
	//int in_channels = dst->in_shape().depth_;
	int out_channels_dst = dst->out_data_shape()[0].depth_;
	int in_channels_dst = dst->in_data_shape()[0].depth_;

	int out_channels = 0;
	int in_channels = 0;
	int window_size = 0;

	fread(&window_size, sizeof(int), 1, if_src);
	fread(&in_channels, sizeof(int), 1, if_src);
	fread(&out_channels, sizeof(int), 1, if_src);
	assert(in_channels_dst == in_channels);
	assert(out_channels_dst == out_channels);

	int dst_idx = 0;
	int src_idx = 0;

	std::cout << window_size << "  " << in_channels << "  " << out_channels<< std::endl;

	vec_t& w = *dst->weights()[0];
	vec_t& b = *dst->weights()[1];
	float *w_i_j = new float;
	// fill weights
	for (int o = 0; o < out_channels; o++) {
		//std::cout <<"o : "<< o << std::endl;
		for (int i = 0; i < in_channels; i++) {
			//std::cout << "i : "<<i << std::endl;

			for (int x = 0; x < window_size * window_size; x++) {
				//TODO
				//dst->weight()[dst_idx++] = weights.data(src_idx++);
				
				fread(w_i_j, sizeof(float), 1, if_src);
				//std::cout << *w_i_j << std::endl;
				w[dst_idx++] = *w_i_j;
			}
		}
	}

	// fill bias
	for (int o = 0; o < out_channels; o++) {
		
		fread(w_i_j, sizeof(float), 1, if_src);
		b[o] = *w_i_j;
	}
	delete w_i_j;
	//std::cout << "delete " << std::endl;
	fclose(if_src);
	//std::cout << "fclose " << std::endl;
	return;
}
inline void load_weights_fullyconnected_fromfile(const std::string src, layer *dst){
	FILE* if_src = fopen(src.c_str(), "rb");
	int dst_out_size = 0;
	int dst_in_size = 0;
	fread(&dst_in_size, sizeof(int), 1, if_src);
	fread(&dst_out_size, sizeof(int), 1, if_src);



	vec_t& w = *dst->weights()[0];
	vec_t& b = *dst->weights()[1];
	float *w_i = new float;
	// fill weights
	for (size_t o = 0; o < dst_out_size; o++) {
		for (size_t i = 0; i < dst_in_size; i++) {
			// TODO: how to access to weights?
			//dst->weight()[i * dst->out_size() + o] = weights.data(curr++); // transpose
			fread(w_i, sizeof(float), 1, if_src);
			w[i + dst_in_size * o] = *w_i; // no transpose
			//w[i * dst_out_size + o] = *w_i; // transpose
		}
	}

	// fill bias
	for (size_t o = 0; o < dst_out_size; o++) {
		// TODO: how to access to biases?
		//dst->bias()[o] = biases.data(o);
		fread(w_i, sizeof(float), 1, if_src);
		b[o] = *w_i;
	}
	delete w_i;
	fclose(if_src);
	return;
}


} // namespace tiny_dnn
