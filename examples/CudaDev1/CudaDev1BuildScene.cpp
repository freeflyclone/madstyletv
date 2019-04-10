/**************************************************************
** CudaDev1BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse, with the framework for CUDA
** API integration in place.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLCuda.h"

XGLCuda::XGLCuda(XGL *px) : pXgl(px) {
	xprintf("XGLCuda::XGLCuda()\n");
	//cuda_main();

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	for (int i = 0; i < dataSize; i++) {
		a[i] = i % 8;
		b[i] = 2;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, sizeof(c));
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, sizeof(a));
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, sizeof(b));
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, sizeof(b), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = LoadKernel(dev_a, dev_b, dev_c, dataSize);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		xprintf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < dataSize; i++)
		xprintf("%d, ", c[i]);
	xprintf("\n");

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}

void XGLCuda::Draw() {
}

void ExampleXGL::BuildScene() {
	XGLCuda *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLCuda(this); return shape; });
}
