#include "XGLCudaInterop.h"

// derived from simpleGL CUDA sample
__global__ void simple_vbo_kernel(XGLRGBA *color, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float u = (float)x / (float)width;
	float v = (float)y / (float)height;
	float freq = 2.0f;

	float w = sinf(u*freq + time) * cosf(v*freq + time) * 127;

	// write output vertex
	color[y * width + x].r = 127 + w;
	color[y * width + x].g = 127 + w;
	color[y * width + x].b = 127 + w;
}

cudaError_t LaunchKernel(XGLRGBA* c, int width, int height, float clock) {
	cudaError_t cudaStatus;

	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(c, width, height, clock);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		xprintf("simple_vbo_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	return cudaSuccess;
Error:
	xprintf("%s: Nope.\n", __FUNCTION__);
	return cudaStatus;
}