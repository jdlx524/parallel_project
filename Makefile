all: single openmp gpu

single:  cloth_sim_openmp.cu
	nvcc cloth_sim_openmp.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_75 -Xcompiler="-march=native" -o single

openmp: cloth_sim_openmp.cu
	nvcc cloth_sim_openmp.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_75 -Xcompiler="-fopenmp  -march=native" -o openmp

gpu: cloth_sim_cuda.cu
	nvcc cloth_sim_cuda.cu -lGL -lGLU -lglut -lGLEW -std=c++11 -O3 -arch=sm_75 -o cloth_sim_cuda
