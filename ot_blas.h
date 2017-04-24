#ifndef OT_BLAS_H
#define OT_BLAS_H

#include <time.h>

extern void ot_sgemm(int m, int n, int k, float alpha,  float* d_A, int lda, float* d_B, int ldb, float beta, float* d_C, int ldc);
extern void ot_dgemm(int m, int n, int k, double alpha,  double* d_A, int lda, double* d_B, int ldb, double beta, double* d_C, int ldc);



#endif
