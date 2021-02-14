#include "XGLMemoryPool.h"

std::mutex XGLCudaMemoryPool::guard;

XGLCudaMemoryPool* XGLCudaMemoryPool::getInstance()
{
	static XGLCudaMemoryPool * instance = NULL;

	if (instance == NULL)
	{
		std::unique_lock<std::mutex> lock(guard);
		if (instance == NULL)
		{
			instance = new XGLCudaMemoryPool();
		}
	}
	return instance;
}

cudaError_t  XGLCudaMemoryPool::cudaMalloc(void ** p, size_t size)
{
	return getInstance()->malloc_d(p, size);
}

cudaError_t XGLCudaMemoryPool::cudaFree(void * p)
{
	return getInstance()->free_d(p);
}

cudaError_t XGLCudaMemoryPool::cudaMallocArray(
	struct cudaArray ** 	array,
	const struct cudaChannelFormatDesc * 	desc,
	size_t 	width,
	size_t 	height,
	unsigned int flags)
{
	return getInstance()->malloc_array(array,
		desc,
		width,
		height,
		flags);
}

cudaError_t XGLCudaMemoryPool::cudaMalloc3DArray(
	struct cudaArray ** 	array,
	const struct cudaChannelFormatDesc * 	desc,
	struct cudaExtent ext,
	unsigned int flags)
{
	return getInstance()->malloc_array_3d(array,
		desc,
		ext,
		flags);
}

cudaError_t XGLCudaMemoryPool::cudaFreeArray(cudaArray * p)
{
	getInstance()->free_array(p);

	return cudaSuccess;
}

cudaError_t XGLCudaMemoryPool::cudaMallocHost(void ** p, size_t size)
{
	return getInstance()->malloc_h(p, size);
}

cudaError_t XGLCudaMemoryPool::cudaHostAlloc(void ** p, size_t size, unsigned int flags)
{
	return getInstance()->hostAlloc_h(p, size, flags);
}

cudaError_t XGLCudaMemoryPool::cudaFreeHost(void * p)
{
	getInstance()->free_h(p);

	return cudaSuccess;
}


cudaError_t XGLCudaMemoryPool::malloc_d(void ** p, size_t size)
{
	int device = 0;
	cudaGetDevice(&device);
	cudaError_t result = cudaSuccess;
	*p = _device.findBlock(size, device);

	if (*p == NULL)
	{
		result = ::cudaMalloc(p, size);
		if (result != cudaSuccess)
		{
			std::cout << "Memory allocation failed: " << result << "\n";
			_device.sweep();
			_array.sweep();
			result = ::cudaMalloc(p, size);
		}
		if (result == cudaSuccess)
			_device.addBlock(*p, size, device);
	}
	return result;
}

cudaError_t XGLCudaMemoryPool::free_d(void * p)
{
	_device.releaseBlock(p);
	return cudaSuccess;
}

cudaError_t XGLCudaMemoryPool::malloc_array(struct cudaArray ** 	array,
	const struct cudaChannelFormatDesc * 	desc,
	size_t 	width,
	size_t 	height,
	unsigned int flags)
{
	int device = 0;
	cudaGetDevice(&device);
	cudaError_t result = cudaSuccess;
	*array = (cudaArray*)_array.findBlock(width, height, 0, *desc, device);

	if (*array == NULL)
	{
		result = ::cudaMallocArray(array, desc, width, height, flags);
		if (result != cudaSuccess)
		{
			std::cout << "Memory allocation failed: " << result << "\n";
			_device.sweep();
			_array.sweep();
			result = ::cudaMallocArray(array, desc, width, height, flags);
		}
		if (result == cudaSuccess)
			_array.addBlock(*array, width, height, 0, *desc, device);
	}
	return result;
}

cudaError_t XGLCudaMemoryPool::malloc_array_3d(struct cudaArray ** 	array,
	const struct cudaChannelFormatDesc * 	desc,
	const struct cudaExtent & ext,
	unsigned int 	flags)
{
	int device = 0;
	cudaGetDevice(&device);
	cudaError_t result = cudaSuccess;
	*array = (cudaArray*)_array.findBlock(ext.width, ext.height, ext.depth, *desc, device);

	if (*array == NULL)
	{
		result = ::cudaMalloc3DArray(array, desc, ext, flags);
		if (result != cudaSuccess)
		{
			std::cout << "Memory allocation failed: " << result << "\n";
			_device.sweep();
			_array.sweep();
			result = ::cudaMalloc3DArray(array, desc, ext, flags);
		}
		if (result == cudaSuccess)
			_array.addBlock(*array, ext.width, ext.height, ext.depth, *desc, device);
	}
	return result;
}

void XGLCudaMemoryPool::free_array(void * p)
{
	_array.releaseBlock(p);
}

cudaError_t XGLCudaMemoryPool::malloc_h(void ** p, size_t size)
{
	int device = 0;
	cudaGetDevice(&device);
	cudaError_t result = cudaSuccess;
	*p = _host.findBlock(size, device);

	if (*p == NULL)
	{
		result = ::cudaMallocHost(p, size);
		if (result != cudaSuccess)
		{
			std::cout << "Memory allocation failed: " << result << "\n";
			_host.sweep();
			result = ::cudaMallocHost(p, size);
		}
		if (result == cudaSuccess)
			_host.addBlock(*p, size, device);
	}
	return result;
}

void XGLCudaMemoryPool::free_h(void * p)
{
	if (!_host.releaseBlock(p))
	{
		_hostAlloc.releaseBlock(p);
	}
}

cudaError_t XGLCudaMemoryPool::hostAlloc_h(void ** p, size_t size, unsigned int flags)
{
	int device = 0;
	cudaGetDevice(&device);
	cudaError_t result = cudaSuccess;
	*p = _hostAlloc.findBlock(size, device);

	if (*p == NULL)
	{
		result = ::cudaHostAlloc(p, size, flags);
		if (result != cudaSuccess)
		{
			std::cout << "Memory allocation failed: " << result << "\n";
			_hostAlloc.sweep();
			result = ::cudaHostAlloc(p, size, flags);
		}
		if (result == cudaSuccess)
			_hostAlloc.addBlock(*p, size, device);
	}
	return result;
}

void XGLCudaMemoryPool::Pool::addBlock(void * ptr, size_t size, int device)
{
	std::unique_lock<std::mutex> lock(_guard);

	_inUse[ptr] = { ptr, size, device };
}

void * XGLCudaMemoryPool::Pool::findBlock(size_t size, int device)
{
	std::unique_lock<std::mutex> lock(_guard);

	for (auto i = _free.begin(); i < _free.end(); ++i)
	{
		if (i->size == size && i->device == device)
		{
			void * p = i->ptr;
			_inUse[p] = *i;
			_free.erase(i);
			return p;
		}
	}
	return NULL;
}

bool XGLCudaMemoryPool::Pool::releaseBlock(void * ptr)
{
	std::unique_lock<std::mutex> lock(_guard);

	auto i = _inUse.find(ptr);

	if (i != _inUse.end())
	{
		_free.push_back(i->second);
		_inUse.erase(i);
		return true;
	}
	return false;
}

void XGLCudaMemoryPool::Pool::sweep()
{
	std::unique_lock<std::mutex> lock(_guard);

	for (auto i = _free.begin(); i < _free.end(); ++i)
	{
		::cudaFree(i->ptr);
	}
	_free.clear();
}

void XGLCudaMemoryPool::ArrayPool::addBlock(void * ptr, size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device)
{
	std::unique_lock<std::mutex> lock(_guard);

	_inUse[ptr] = { ptr, width, height, depth, desc, device };
}

void * XGLCudaMemoryPool::ArrayPool::findBlock(size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device)
{
	std::unique_lock<std::mutex> lock(_guard);

	for (auto i = _free.begin(); i < _free.end(); ++i)
	{
		if (i->width == width && i->height == height && i->depth == depth && i->desc.x == desc.x && i->desc.y == desc.y && i->desc.z == desc.z && i->desc.w == desc.w &&  i->desc.f == desc.f && i->device == device)
		{
			void * p = i->ptr;
			_inUse[p] = *i;
			_free.erase(i);
			return p;
		}
	}
	return NULL;
}

bool XGLCudaMemoryPool::ArrayPool::releaseBlock(void * ptr)
{
	std::unique_lock<std::mutex> lock(_guard);

	auto i = _inUse.find(ptr);

	if (i != _inUse.end())
	{

		_free.push_back(i->second);

		_inUse.erase(i);

		return true;
	}
	return false;
}

void XGLCudaMemoryPool::ArrayPool::sweep()
{
	std::unique_lock<std::mutex> lock(_guard);

	for (auto i = _free.begin(); i < _free.end(); ++i)
	{
		::cudaFree(i->ptr);
	}
	_free.clear();
}