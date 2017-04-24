CC = gcc
ICC = icc
GPU_CC = nvcc

CC_FLAGS = 
ICC_FLAGS = 
GPU_CC_FLAGS = -Wno-deprecated-gpu-targets

GPU_ROOT = /usr/local/cuda-8.0
GPU_LIB_ROOT = $(GPU_ROOT)/lib64
GPU_LIBS = -lcudart -lcublas -lcurand -lcudadevrt
GPU_INCLUDE = $(GPU_ROOT)/include

CPU_ROOT = /opt/intel/compilers_and_libraries_2017.1.132/linux/mkl
CPU_LIB_ROOT = $(CPU_ROOT)/lib/intel64/
CPU_LIBS = -lrt -lmkl_intel_lp64 -lmkl_core -lgomp -lmkl_gnu_thread -lpthread -lm -ldl
CPU_INCLUDE = $(CPU_LIB_ROOT)/include

LD_CPU_LIBS = $(CPU_LIBS) -L$(CPU_LIB_ROOT) -I$(CPU_INCLUDE)
LD_GPU_LIBS =  $(GPU_LIBS) -L$(GPU_LIB_ROOT) -I$(GPU_INCLUDE)


all: ot_blas 

##############################################################
sum: sum.a
	gcc sum_driver.c -o sum -lsum -L. 

sum.a: sum.o
	ar rcs libsum.a sum.o

sum.o:
	gcc -c sum.c -o sum.o


##############################################################	
ot_blas: gpublas.a cpu_blas.a ot_blas.o
	g++ ot_blas.o -o ot_blas -lgpublas -lcpublas -L. $(LD_GPU_LIBS) $(LD_CPU_LIBS)
	
ot_blas.o:
	g++ -c ot_blas.c -o ot_blas.o -std=c11 $(LD_GPU_LIBS) $(LD_CPU_LIBS)
	
##############################################################
cpu_blas.a: cpu_blas.o
	ar rcs libcpublas.a cpu_blas.o

cpu_blas.o: 
	icpc -c cpu_blas.c -o cpu_blas.o -lmkl

	
##############################################################
	
gpublas.a: link.o
	nvcc --lib --output-file libgpublas.a gpu_blas.o link.o -Wno-deprecated-gpu-targets
	
link.o: gpu_blas.o
	nvcc --gpu-architecture=sm_20 --device-link gpu_blas.o --output-file link.o -Wno-deprecated-gpu-targets

gpu_blas.o:
	nvcc --gpu-architecture=sm_20 --device-c gpu_blas.cu -lcublas -lcurand -Wno-deprecated-gpu-targets

##############################################################

clean:
	rm -rf *.o *.gpu *.a *.ptx *.stub.c
