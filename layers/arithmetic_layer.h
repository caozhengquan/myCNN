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
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/**
 * element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```
 **/
class elementwise_add_layer : public layer {
public:
    /**
     * @param num_args [in] number of inputs
     * @param dim      [in] number of elements for each input
     */
    elementwise_add_layer(cnn_size_t num_args, cnn_size_t dim)
    : layer(std::vector<vector_type>(num_args, vector_type::data), {vector_type::data}), num_args_(num_args), dim_(dim) {}

    std::string layer_type() const override {
        return "elementwise-add";
    }

    std::vector<shape3d> in_shape() const override {
        return std::vector<shape3d>(num_args_, shape3d(dim_,1,1));
    }

    std::vector<shape3d> out_shape() const override {
        return{ shape3d(dim_,1,1) };
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>& out_data) override {
        const tensor_t& in1 = *in_data[0];
        tensor_t& out = *out_data[0];

        out = in1;

        // @todo parallelize
        for (size_t sample = 0; sample < in1.size(); ++sample) {
            for (cnn_size_t i = 1; i < num_args_; i++) {
                std::transform((*in_data[i])[sample].begin(),
                               (*in_data[i])[sample].end(),
                               out[sample].begin(),
                               out[sample].begin(),
                               [](float_t x, float_t y){ return x + y; });
            }
        }
    }


    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<elementwise_add_layer> & construct) {
        cnn_size_t num_args, dim;

        ar(cereal::make_nvp("num_args", num_args), cereal::make_nvp("dim", dim));
        construct(num_args, dim);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("num_args", num_args_), cereal::make_nvp("dim", dim_));
    }
private:
    cnn_size_t num_args_;
    cnn_size_t dim_;
};

class elementwise_max_layer : public layer {
public:
	/**
	* @param num_args [in] number of inputs
	* @param dim      [in] number of elements for each input
	*/
	elementwise_max_layer(cnn_size_t num_args, cnn_size_t dim)
		: layer(std::vector<vector_type>(num_args, vector_type::data), { vector_type::data }), num_args_(num_args), dim_(dim) {}

	std::string layer_type() const override {
		return "elementwise-max";
	}

	std::vector<shape3d> in_shape() const override {
		return std::vector<shape3d>(num_args_, shape3d(dim_, 1, 1));
	}

	std::vector<shape3d> out_shape() const override {
		return{ shape3d(dim_, 1, 1) };
	}

	void forward_propagation(const std::vector<tensor_t*>& in_data,
		std::vector<tensor_t*>& out_data) override {
		double st1 = clock();
		if (num_args_ == 2)
		{
			tensor_t& in1_t = *in_data[0];
			tensor_t& in2_t = *in_data[1];
			tensor_t& out_t = *out_data[0];
			vec_t& in1 = in1_t[0];
			vec_t& in2 = in2_t[0];
			vec_t& out = out_t[0];
			out = in1;
			//std::cout <<"max_elt.size()= "<< in1.size() << std::endl;;
			for (size_t i = 0; i < in1.size();i++)
			{
				if (in1[i] < in2[i])
				{
					out[i] = in2[i];
				}
			}
		}
		else
		{
			const tensor_t& in1 = *in_data[0];
			tensor_t& out = *out_data[0];

			out = in1;

			// @todo parallelize
			for (size_t sample = 0; sample < in1.size(); ++sample) {
				for (cnn_size_t i = 1; i < num_args_; i++) {
					std::transform((*in_data[i])[sample].begin(),
						(*in_data[i])[sample].end(),
						out[sample].begin(),
						out[sample].begin(),
						[](float_t x, float_t y){ return (x>y)?x:y; });
				}
			}
		}
		double st2 = clock();
		//std::cout << "max_elt elapse : " << st2 - st1 << std::endl;
	}

	

	template <class Archive>
	static void load_and_construct(Archive & ar, cereal::construct<elementwise_max_layer> & construct) {
		cnn_size_t num_args, dim;

		ar(cereal::make_nvp("num_args", num_args), cereal::make_nvp("dim", dim));
		construct(num_args, dim);
	}

	template <class Archive>
	void serialize(Archive & ar) {
		layer::serialize_prolog(ar);
		ar(cereal::make_nvp("num_args", num_args_), cereal::make_nvp("dim", dim_));
	}
private:
	cnn_size_t num_args_;
	cnn_size_t dim_;
};


} // namespace tiny_dnn

