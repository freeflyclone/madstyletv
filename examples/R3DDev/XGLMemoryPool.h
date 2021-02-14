#ifndef XGLGPUMemoryPool_h
#define XGLGPUMemoryPool_h

#include <R3DSDK.h>
#include <R3DSDKCuda.h>
#include <R3DSDKDefinitions.h>

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mutex>
#include <thread>
#include <condition_variable>
#include <list>
#include <map>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <time.h>
#include <iostream>

#define PER_GPU_QUEUE_SIZE	4
#define TOTAL_FRAMES		1000
#define GPU_THREADS			1		// set to 2 for 2 GPUs, etc.
#define NUM_STREAMS			4
#define FRAME_QUEUE_SIZE	(PER_GPU_QUEUE_SIZE * GPU_THREADS)

class XGLCudaMemoryPool
{
public:
	static XGLCudaMemoryPool* getInstance();
	static cudaError_t cudaMalloc(void** p, size_t size);
	static cudaError_t cudaFree(void* p);

	static cudaError_t cudaMallocArray(
		struct cudaArray** array,
		const struct cudaChannelFormatDesc* desc,
		size_t width,
		size_t height = 0,
		unsigned int flags = 0);

	static cudaError_t cudaMalloc3DArray(
		struct cudaArray** array,
		const struct cudaChannelFormatDesc* desc,
		struct cudaExtent ext,
		unsigned int flags = 0);

	static cudaError_t cudaFreeArray(cudaArray* p);
	static cudaError_t cudaMallocHost(void** p, size_t size);
	static cudaError_t cudaHostAlloc(void** p, size_t size, unsigned int flags);
	static cudaError_t cudaFreeHost(void* p);

private:
	static std::mutex guard;

	cudaError_t malloc_d(void ** p, size_t size);
	cudaError_t free_d(void * p);
	cudaError_t malloc_array(struct cudaArray** array,
		const struct cudaChannelFormatDesc* desc,
		size_t width,
		size_t height = 0,
		unsigned int flags = 0);

	cudaError_t malloc_array_3d(struct cudaArray** array,
		const struct cudaChannelFormatDesc*	desc,
		const struct cudaExtent & ext,
		unsigned int 	flags = 0);

	void free_array(void * p);
	cudaError_t malloc_h(void ** p, size_t size);
	void free_h(void * p);
	cudaError_t hostAlloc_h(void ** p, size_t size, unsigned int flags);

	struct BLOCK
	{
		void * ptr;
		size_t size;
		int device;
	};

	struct ARRAY
	{
		void * ptr;
		size_t width;
		size_t height;
		size_t depth;
		cudaChannelFormatDesc desc;
		int device;
	};

	class Pool
	{
	public:
		void addBlock(void * ptr, size_t size, int device);
		void * findBlock(size_t size, int device);
		bool releaseBlock(void * ptr);
		void sweep();
	private:
		std::map<void*, BLOCK> _inUse;
		std::vector<BLOCK> _free;
		std::mutex _guard;
	};

	class ArrayPool
	{
	public:
		void addBlock(void * ptr, size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device);
		void* findBlock(size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device);
		bool releaseBlock(void * ptr);
		void sweep();

	private:
		std::map<void*, ARRAY> _inUse;
		std::vector<ARRAY> _free;
		std::mutex _guard;
	};

	Pool _device;
	Pool _host;
	Pool _hostAlloc;
	ArrayPool _array;
};

#endif