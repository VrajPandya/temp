#include<stdio.h>
#include"cpu_blas.h"
#include<mkl.h>

void cpu_sgemm(int m, int n, int k, float alpha, float* h_A, int lda,
		float* h_B, int ldb, float beta, float* h_C, int ldc) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, h_A, k, h_B, n, beta, h_C, n);


}

void cpu_dgemm(int m, int n, int k, double alpha, double* h_A, int lda,
		double* h_B, int ldb, double beta, double* h_C, int ldc) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, h_A, k, h_B, n, beta, h_C, n);


}
