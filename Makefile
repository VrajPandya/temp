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

##############################################################
sum: sum.a
	gcc sum_driver.c -o sum -lsum -L. 

sum.a: sum.o
	ar rcs libsum.a sum.o

sum.o:
	gcc -c sum.c -o sum.o


##############################################################	
ot_blas: gpublas.a ot_blas.o
	g++ ot_blas.o -o ot_blas -lgpublas -L. -lcudart -lcudadevrt -lcublas -L/usr/local/cuda-8.0/lib64
	
ot_blas.o:
	g++ -c ot_blas.c -o ot_blas.o -std=c11
	
gpublas.a: link.o
	nvcc --lib --output-file libgpublas.a gpu_blas.o link.o -Wno-deprecated-gpu-targets
	
link.o: gpu_blas.o
	nvcc --gpu-architecture=sm_20 --device-link gpu_blas.o --output-file link.o -Wno-deprecated-gpu-targets

gpu_blas.o:
	nvcc --gpu-architecture=sm_20 --device-c gpu_blas.cu -lcublas -lcurand -Wno-deprecated-gpu-targets

clean:
	rm -rf *.o *.gpu *.a *.ptx *.stub.c
