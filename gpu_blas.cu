#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "gpu_blas.h"

void gpu_sgemm(int m, int n, int k, float alpha, float* h_A, int lda,
		float* h_B, int ldb, float beta, float* h_C, int ldc) {
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Fill the arrays A and B on GPU with random numbers
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, m * n * sizeof(float));
	cudaMalloc(&d_B, n * k * sizeof(float));
	cudaMalloc(&d_C, m * k * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, n * m * sizeof(float), cudaMemcpyHostToDevice);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda,
			d_B, ldb, &beta, d_C, ldc);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

	// Destroy the handle
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
