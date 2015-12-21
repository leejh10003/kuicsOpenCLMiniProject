edit: vectorAddition.out

vectorAddition.out: main.cpp
	g++ -I $(CUDA_HOME)/include/ -L $(CUDA_HOME)/lib64 main.cpp -o vectorAddition.out -lOpenCL
