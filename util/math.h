#pragma once
#include <omp.h>
#include <iostream>
#include <pthread.h>
struct union_mm {
	int st_m;
	int ed_m;
	int n;
	int k;
	const float* A ;
	const float* B ;
	float* C;
	union_mm(int st_m,int ed_m,int n,int k,const float *A,const float *B,float *C){
		this->st_m = st_m;
		this->ed_m = ed_m;
		this->n = n;
		this->k = k;

		this->A = A;
		this->B = B;
		this->C = C;
	
	}
};
namespace tiny_dnn {
		inline float simd_dot(const float* x, const float* y, const long& len) {
		  float inner_prod = 0.0f;
		  __m128 X, Y; // 128-bit values
		  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
		  float temp[4];

		  long i;
		  for (i = 0; i + 4 < len; i += 4) {
			  X = _mm_loadu_ps(x + i); // load chunk of 4 floats
			  Y = _mm_loadu_ps(y + i);
			  acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
		  }
		  _mm_storeu_ps(&temp[0], acc); // store acc into an array
		  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

		  // add the remaining values
		  for (; i < len; ++i) {
			  inner_prod += x[i] * y[i];
		  }
		  return inner_prod;
		}

		inline void* multi_product(void *arg)
		{
			union_mm* in = (union_mm *)arg;
			int st_m = in->st_m;;
			int ed_m = in->ed_m;
			int n = in->n;
			int k = in->k;
			const float* A = in->A;
			const float* B = in->B; 
			float* C = in->C;
			//std::cout << "n: " << n << std::endl;
			const float* x = B+st_m*k;
			for (int i = st_m, idx = st_m*n ; i < ed_m; ++i) {
				const float* y = A;
				for (int j = 0; j < n; ++j, ++idx) {
					C[idx] = simd_dot(x, y, k);
					y += k;
				}
				x += k;
			}

			return (void*)0;
		}


		inline void matrix_procuct(const float* A, const float* B, float* C, const int n,
			const int m, const int k, bool ta, bool tb) {
		#if 0
			assert(ta && !tb);
			const float* x = B;
			float res = 0;
			for (int i = 0; i < m; i++) {
				const float* y = A;
			#pragma omp parallel for
				for (int j = 0; j < n;j++) {
					
					C[i*n + j]= simd_dot(x + i*k, y + j*k, k);
					//res = C[i*n + j];

					//C[i*n + j]=res;
					//std::cout << "res : " << res << std::endl;
				}
			}
		#else
#if 1
			assert(ta && !tb);
			#define NUM_THREAD 4
			pthread_t id[NUM_THREAD];
			union_mm** mm = new union_mm*[NUM_THREAD];
			for (int i = 0; i < NUM_THREAD; i++)
			{
				mm[i] = new union_mm(i*m / NUM_THREAD, (i+1)*m / NUM_THREAD, n, k, A, B, C);
			}
			for (int i = 0; i < NUM_THREAD; i++)
			{
				pthread_create(&id[i], NULL, multi_product, (void*)mm[i]);
			}
			for (int i = 0; i < NUM_THREAD; i++)
			{
				pthread_join(id[i], NULL);
				delete mm[i];
			}
			delete[]mm;			
#else

			assert(ta && !tb);
		  const float* x = B;
		  for (int i = 0, idx = 0; i < m; ++i) {
			const float* y = A;
			for (int j = 0; j < n; ++j, ++idx) {
			  C[idx] =simd_dot(x, y, k);
			  y += k;
			}
			x += k;
		  }
#endif
		#endif
		}
}
