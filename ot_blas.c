#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>

#include "gpu_blas.h"
#include "cpu_blas.h"
#include "ot_blas.h"

#define RAND 0

#define MAX_double 50.0

#define PRINT 1

#define MY_MY_PERF 0

//int cpu_ops;

double second(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
}

// Fill the array A(rows_A, cols_A) with random numbers on GPU
void fill_rand(double *A, int rows, int cols) {
	// Create a pseudo-random number generator

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[(i * rows) + j] = rand() / MAX_double;
		}
	}
}

void fill_seq(double *A, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[(i * rows) + j] = (i * rows) + j;
		}
	}
}

void diff_mat(double* diff, double* A, double* B, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			diff[(i * rows) + j] = A[(i * rows) + j] - B[(i * rows) + j];
		}
	}
}

//Print matrix A(rows_A, cols_A) storage in row-major format
void print_matrix(const double *A, int rows_A, int cols_A, int lda) {

	for (int i = 0; i < (rows_A * cols_A) / lda; ++i) {
		for (int j = 0; j < lda; ++j) {
			printf("%f ", A[(i * lda) + j]);
		}
		printf("\n");
	}
}

void ot_dgemm(int m, int n, int k, double alpha, double* h_A, int lda,
		double* h_B, int ldb, double beta, double* h_C, int ldc) {
	int cpu_rows = 1;
	int cpu_ops = cpu_rows * lda;

	int c = cpu_ops;

	cublasInit();
	cublasHandle_t handle;
	cublasCreate(&handle);

	double *d_A, *d_B, *d_C, *temp;
	cudaMalloc(&d_A, (m) * k * sizeof(double));
	cudaMalloc(&d_B, n * k * sizeof(double));
	cudaMalloc(&d_C, (m) * n * sizeof(double));


	temp = (double *)malloc((m - cpu_rows) * k * sizeof(double));
	memset(temp, 0 , (m - cpu_rows) * k * sizeof(double));
	//print_matrix(h_A + cpu_ops, m - cpu_rows, k , k);
	memcpy(temp, h_A + (cpu_ops), ((m - cpu_rows) * k) * sizeof(double));
	print_matrix(temp, m - cpu_rows, k , k);



	cudaMemcpyAsync(d_A, h_A , (m * k) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_B, h_B, n * k * sizeof(double), cudaMemcpyHostToDevice);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - cpu_rows, n, k, &alpha, d_A + cpu_ops, lda, d_B, ldb, &beta, d_C + cpu_ops, ldc);

	cudaMemcpyAsync(h_C + (cpu_ops), d_C + cpu_ops, ((m - cpu_rows) * n) * sizeof(double),
			cudaMemcpyDeviceToHost);


	cpu_dgemm(cpu_rows, n, k, alpha, h_A, lda, h_B, ldb, beta, h_C, ldc);
	//cpu_dgemm(m - cpu_rows, n, k, alpha, h_A + cpu_ops, lda, h_B, ldb, beta, h_C + cpu_ops, ldc);
	// Destroy the handle

	cudaDeviceSynchronize();
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(temp);

}

int main(int argc, const char** argv) {
	// Allocate 3 arrays on CPU
	int rows_A, cols_A, rows_B, cols_B, rows_C, cols_C;
	float flopsCoef = 2.0;
	int size = 5;

	double gpu_start, gpu_stop, cpu_start, cpu_stop, ot_start, ot_stop;
	int cpu_N = 1, gpu_N = 1;
	// for simplicity we are going to use square arrays
	if (argc > 1) {
		size = atoi(argv[1]);
	}
	rows_A = cols_A = rows_B = cols_B = rows_C = cols_C = size;

	double *h_A = (double *) malloc(rows_A * cols_A * sizeof(double));
	double *h_B = (double *) malloc(rows_B * cols_B * sizeof(double));
	double *h_gpu_C = (double *) malloc(rows_C * cols_C * sizeof(double));
	double *h_cpu_C = (double *) malloc(rows_C * cols_C * sizeof(double));
	double *ot_C = (double *) malloc(rows_C * cols_C * sizeof(double));
	double *h_diff_C = (double *) malloc(rows_C * cols_C * sizeof(double));
	// Allocate 3 arrays on GPU

	srand(time(NULL));

	memset(h_gpu_C, 0, rows_C * cols_C * sizeof(double));
	memset(h_cpu_C, 0, rows_C * cols_C * sizeof(double));
	memset(ot_C, 0, rows_C * cols_C * sizeof(double));
	memset(h_diff_C, 0, rows_C * cols_C * sizeof(double));

#if RAND
	fill_rand(h_A, rows_A, cols_A);
	fill_rand(h_B, rows_B, cols_B);
#else
	fill_seq(h_A, rows_A, cols_A);
	fill_seq(h_B, rows_B, cols_B);
#endif
	// Multiply A and B on GPU
#if MY_PERF

	gpu_start = second();

#endif
	gpu_dgemm(rows_A, cols_A, cols_B, 1.0, h_A, cols_A, h_B, cols_B, 0.0,
			h_gpu_C, cols_C);
#if MY_PERF
	gpu_stop = cpu_start = second();
	// Multiply A and B on CPU
#endif
	cpu_dgemm(rows_A, cols_A, cols_B, 1.0, h_A, cols_A, h_B, cols_B, 0.0,
			h_cpu_C, cols_C);
#if MY_PERF
	cpu_stop = second();
#endif
	//both CPU and GPU execution
	ot_dgemm(rows_A, cols_A, cols_B, 1.0, h_A, cols_A, h_B, cols_B, 0.0, ot_C,
			cols_C);

#if PRINT
	printf("++++++++++++++++++++++++++A++++++++++++++++++++++++++ :\n");
	print_matrix(h_A, rows_A, cols_A, cols_A);

	printf("++++++++++++++++++++++++++B++++++++++++++++++++++++++ :\n");
	print_matrix(h_B, rows_B, cols_B, cols_B);

	printf("++++++++++++++++++++++++++gpu C++++++++++++++++++++++++++ :\n");
	print_matrix(h_gpu_C, rows_C, cols_C, cols_C);

	printf("++++++++++++++++++++++++++cpu C++++++++++++++++++++++++++:\n");
	print_matrix(h_cpu_C, rows_B, cols_C, cols_C);
	// Free CPU memory

	printf("++++++++++++++++++++++++++ot C++++++++++++++++++++++++++:\n");
	print_matrix(ot_C, rows_B, cols_C, cols_C);

	diff_mat(h_diff_C, h_cpu_C, ot_C, rows_B, cols_C);

	printf("++++++++++++++++++++++++++diff++++++++++++++++++++++++++:\n");
	print_matrix(h_diff_C, rows_B, cols_C, cols_C);
#endif

#if MY_PERF
	printf("^^^^ elapsed for GPU = %10.8f sec  GFLOPS=%g\n",
			(gpu_stop - gpu_start),
			cpu_N * (1e-9 * flopsCoef * rows_A * cols_A * cols_B)
			/ (gpu_stop - gpu_start));
	printf("^^^^ elapsed for CPU = %10.8f sec  GFLOPS=%g\n",
			(cpu_stop - cpu_start),
			cpu_N * (1e-9 * flopsCoef * rows_A * cols_A * cols_B)
			/ (cpu_stop - cpu_start));
#endif

	free(h_A);
	free(h_B);
	free(h_gpu_C);
	free(h_cpu_C);

	return 0;
}
