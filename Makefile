CC = gcc
ICC = icc
GPU_CC = nvcc

CC_FLAGS = 
ICC_FLAGS = 
GPU_CC_FLAGS = -Wno-deprecated-gpu-targets

GPU_ROOT = /usr/local/cuda-8.0
GPU_LIB_ROOT = $(GPU_ROOT)/lib64
GPU_LIBS = -lcudart -lcublas -lcurand
GPU_INCLUDE = $(GPU_ROOT)/include

CPU_LIBS =
INTEL_LIB =  


all: ot_blas 
	gcc ot_blas.o gpu_blas.o -o ot_blas -std=c11

ot_blas: gpu_blas
	gcc -c ot_blas.c -o ot_blas.o -std=c11

gpu_blas:
	nvcc --device-c gpu_blas.cu -o gpu_blas.o -lcublas -lcurand -Wno-deprecated-gpu-targets
	
