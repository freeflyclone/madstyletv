#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "xutils.h"

// this is a CUDA kernel that adds "a" to "b" with results in "c"
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

cudaError_t LoadKernel(int *dev_a, int *dev_b, int *dev_c, int n) {
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, n>>> (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		xprintf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	return cudaSuccess;
Error:
	xprintf("%s: Nope.\n", __FUNCTION__);
	return cudaStatus;
}