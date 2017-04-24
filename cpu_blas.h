#ifndef CPU_BLAS_H
#define CPU_BLAS_H

extern void cpu_sgemm(int m,int n, int k, float alpha,  float* d_A, int lda, float* d_B, int ldb, float beta, float* d_C, int ldc);

extern void cpu_dgemm(int m,int n, int k, double alpha,  double* d_A, int lda, double* d_B, int ldb, double beta, double* d_C, int ldc);

#endif
