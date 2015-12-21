#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#pragma warning(disable: 4996)
#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
	cl_int ret;
	// Create the two input vectors
	int i;
	const int LIST_SIZE = 1024;
	int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
	int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	// Load the kernel source code into the array source_str
	std::ifstream sourceFile("vector_add_kernel.cl");
	std::string sourceString((std::istreambuf_iterator<char>(sourceFile)),
		(std::istreambuf_iterator<char>()));
	sourceFile.close();

	int sourceLength = sourceString.length();
	char* sourceConverted = (char*)malloc((sourceLength + 1) * sizeof(char));
	strcpy(sourceConverted,
		sourceString.c_str());

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	ret = clGetPlatformIDs(1,
		&platform_id,
		&ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL,
		1,
		&device_id,
		NULL,
		NULL,
		&ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context,
		device_id,
		0,
		&ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int),
		NULL,
		&ret);
	cl_mem b_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int),
		NULL,
		&ret);
	cl_mem c_mem_obj = clCreateBuffer(context,
		CL_MEM_WRITE_ONLY,
		LIST_SIZE * sizeof(int),
		NULL,
		&ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue,
		a_mem_obj,
		CL_TRUE,
		0,
		LIST_SIZE * sizeof(int),
		A,
		0,
		NULL,
		NULL);
	ret = clEnqueueWriteBuffer(command_queue,
		b_mem_obj,
		CL_TRUE,
		0,
		LIST_SIZE * sizeof(int),
		B,
		0,
		NULL,
		NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context,
		1,
		(const char **)&sourceConverted,
		(const size_t *)&sourceLength,
		&ret);

	// Build the program
	ret = clBuildProgram(program,
		1,
		&device_id,
		NULL,
		NULL,
		NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel,
		0,
		sizeof(cl_mem),
		(void *)&a_mem_obj);
	ret = clSetKernelArg(kernel,
		1,
		sizeof(cl_mem),
		(void *)&b_mem_obj);
	ret = clSetKernelArg(kernel,
		2,
		sizeof(cl_mem),
		(void *)&c_mem_obj);

	// Execute the OpenCL kernel on the list
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue,
		kernel,
		1,
		NULL,
		&global_item_size,
		&local_item_size,
		0,
		NULL,
		NULL);

	// Read the memory buffer C on the device to the local variable C
	int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
	ret = clEnqueueReadBuffer(command_queue,
		c_mem_obj,
		CL_TRUE,
		0,
		LIST_SIZE * sizeof(int),
		C,
		0,
		NULL,
		NULL);

	// Display the result to the screen
	for (i = 0; i < LIST_SIZE; i++)
		std::cout<< A[i] << " + " << B[i] << " = " << C[i] << std::endl;

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	system("pause");
	return 0;
}
