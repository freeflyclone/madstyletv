#include "XGLCudaInterop.h"

// derived from simpleGL CUDA sample
__global__ void simple_vbo_kernel(XGLVertexAttributes *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)height;
	float v = y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x].v.x = u;
	pos[y*width + x].v.y = v;
	pos[y*width + x].v.z = w;
}

cudaError_t LaunchKernel(XGLVertexAttributes* v, int width, int height, float clock) {
	cudaError_t cudaStatus;

	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(v, width, height, clock);

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