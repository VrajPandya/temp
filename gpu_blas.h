#ifndef GPU_BLAS_H
#define GPU_BLAS_H

extern void gpu_sgemm(int m,int n, int k, float alpha,  float* d_A, int lda, float* d_B, int ldb, float beta, float* d_C, int ldc);

#endif
