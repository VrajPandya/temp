#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu_blas.h"
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
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;

	float *h_A = (float *) malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *) malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *) malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU


	fill_rand(h_A, nr_rows_A, nr_cols_A);
	fill_rand(h_B, nr_rows_B, nr_cols_B);



	// Multiply A and B on GPU
	gpu_sgemm(nr_rows_A, nr_cols_A, nr_cols_B, 1.0,  h_A, nr_rows_A, h_B, nr_cols_B, 0.0, h_C, nr_cols_B);



	printf("A:\n");
	print_matrix(h_A, nr_rows_A, nr_cols_A);

	printf("B:\n");
	print_matrix(h_B, nr_rows_B, nr_cols_B);

	printf("C:\n");
	print_matrix(h_C, nr_rows_C, nr_cols_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
