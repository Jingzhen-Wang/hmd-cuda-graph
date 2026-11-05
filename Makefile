CC      ?= mpicc
NVCC    ?= nvcc
CFLAGS  = -O -Iinclude -fopenmp
NVFLAGS = -O -Iinclude -ccbin mpicxx -Xcompiler -fopenmp

.PHONY: all gpu graph clean

all: hmd_cpu

hmd_cpu: src/hmd_cpu.c include/hmd.h
	$(CC) $(CFLAGS) -o hmd_cpu src/hmd_cpu.c -lm

gpu: src/hmd_gpu.cu include/hmd.h
	$(NVCC) $(NVFLAGS) -o hmd_gpu src/hmd_gpu.cu -lm

graph: src/hmd_graph.cu include/hmd.h
	$(NVCC) $(NVFLAGS) -o hmd_graph src/hmd_graph.cu -lm

clean:
	rm -f hmd_cpu hmd_gpu hmd_graph *.o
