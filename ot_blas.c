#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "gpu_blas.h"
#include "cpu_blas.h"

#define RAND 0

#define MAX_FLOAT 50.0

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void fill_rand(float *A, int rows, int cols) {
	// Create a pseudo-random number generator

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[(i * rows) + j] = rand() / MAX_FLOAT;
		}
	}
}

void fill_seq(float *A, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[(i * rows) + j] = (i * rows) + j;
		}
	}
}

void diff_mat(float* diff, float* A, float* B, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			diff[(i * rows) + j] = A[(i * rows) + j] - B[(i * rows) + j];
		}
	}
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in row-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			printf("%f ", A[i * nr_rows_A + j]);
		}
		printf("\n");
	}
}

int main() {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 5;

	float *h_A = (float *) malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *) malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_gpu_C = (float *) malloc(nr_rows_C * nr_cols_C * sizeof(float));
	float *h_cpu_C = (float *) malloc(nr_rows_C * nr_cols_C * sizeof(float));
	float *h_diff_C = (float *) malloc(nr_rows_C * nr_cols_C * sizeof(float));
	// Allocate 3 arrays on GPU

	srand(time(NULL));

	memset(h_gpu_C, 0, nr_rows_C * nr_cols_C * sizeof(float));
	memset(h_cpu_C, 0, nr_rows_C * nr_cols_C * sizeof(float));
	memset(h_diff_C, 0, nr_rows_C * nr_cols_C * sizeof(float));

#if RAND
	fill_rand(h_A, nr_rows_A, nr_cols_A);
	fill_rand(h_B, nr_rows_B, nr_cols_B);
#else
	fill_seq(h_A, nr_rows_A, nr_cols_A);
	fill_seq(h_B, nr_rows_B, nr_cols_B);
#endif
	// Multiply A and B on GPU
	gpu_sgemm(nr_rows_A, nr_cols_A, nr_cols_B, 1.0, h_A, nr_rows_A, h_B,
			nr_cols_B, 0.0, h_gpu_C, nr_cols_B);
	// Multiply A and B on CPU
	cpu_sgemm(nr_rows_A, nr_cols_A, nr_cols_B, 1.0, h_A, nr_rows_A, h_B,
			nr_cols_B, 0.0, h_cpu_C, nr_cols_B);

	printf("A:\n");
	print_matrix(h_A, nr_rows_A, nr_cols_A);

	printf("B:\n");
	print_matrix(h_B, nr_rows_B, nr_cols_B);

	printf("gpu C:\n");
	print_matrix(h_gpu_C, nr_rows_C, nr_cols_C);

	printf("cpu C:\n");
	print_matrix(h_cpu_C, nr_rows_C, nr_cols_C);
	// Free CPU memory
	diff_mat(h_diff_C, h_cpu_C, h_gpu_C, nr_rows_C, nr_cols_C);

	printf("diff:\n");
	print_matrix(h_diff_C, nr_rows_C, nr_cols_C);
	free(h_A);
	free(h_B);
	free(h_gpu_C);
	free(h_cpu_C);

	return 0;
}
