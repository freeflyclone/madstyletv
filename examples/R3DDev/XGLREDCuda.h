#ifndef XGLREDCUDA_H
#define XGLREDCUDA_H

#include "XGL.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string.h>
#include <stdio.h>

#include <R3DSDK.h>
#include <R3DSDKCuda.h>
#include <R3DSDKDefinitions.h>

#include "XGLMemoryPool.h"

// max expected dimensions for known RED sensors currently on the planet
const int defaultCudaTexWidth = 8192;
const int defaultCudaTexHeight = 4320;

struct XGLREDCudaRGB16 {
	uint16_t r, g, b;
};

class XGLREDCuda : public XGLTexQuad
{
public:
	typedef std::function<void(R3DSDK::AsyncDecompressJob*)> CompletionFunc;
	typedef std::vector<CompletionFunc> CompletionFuncs;

	XGLREDCuda(std::string clipName);
	void StartVideoDecode(int frame);

	void AllocatePBOOutputBuffer();
	void GenR3DInterleavedTextureBuffer(const int width, const int height);
	void Draw();

	static void getCurrentTimestamp();

	R3DSDK::DebayerCudaJob* DebayerAllocate(
		const R3DSDK::AsyncDecompressJob* job,
		R3DSDK::ImageProcessingSettings* imageProcessingSettings,
		R3DSDK::VideoPixelType pixelType);

	void DebayerFree(R3DSDK::DebayerCudaJob * job);

	template<typename T> class ConcurrentQueue
	{
	private:
		std::mutex QUEUE_MUTEX;
		std::condition_variable QUEUE_CV;
		std::list<T *> QUEUE;

	public:
		void push(T * job)
		{
			std::unique_lock<std::mutex> lck(QUEUE_MUTEX);
			QUEUE.push_back(job);
			QUEUE_CV.notify_all();
		}

		void pop(T * & job)
		{
			std::unique_lock<std::mutex> lck(QUEUE_MUTEX);

			while (QUEUE.size() == 0)
				QUEUE_CV.wait(lck);

			job = QUEUE.front();
			QUEUE.pop_front();
		}

		size_t size() const
		{
			return QUEUE.size();
		}
	};

	ConcurrentQueue<R3DSDK::AsyncDecompressJob> JobQueue;
	ConcurrentQueue<R3DSDK::AsyncDecompressJob> CompletionQueue;

	void CompletionThread();
	void GpuThread(int device);
	static void CpuCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus);
	unsigned char* AlignedMalloc(size_t & sizeNeeded);
	void AddCompletionFunction(XGLREDCuda::CompletionFunc fn);

public:
	std::string clipName;
	R3DSDK::REDCuda* m_pREDCuda{ nullptr };
	R3DSDK::GpuDecoder* m_pGpuDecoder{ nullptr };

	R3DSDK::REDCuda * OpenCuda(int & deviceId);

	volatile bool firstFrameDecoded = false;

	void FirstFrameCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus);

	int CUDA_DEVICE_ID = 0;
	volatile int cpuDone = 0;
	volatile int gpuDone = 0;
	CompletionFuncs completionFuncs;

	GLuint pbo;
	cudaGraphicsResource *cudaPboResource;
	void* pPboCudaMemory{ nullptr };
	size_t pboCudaMemorySize{ 0 };
	size_t m_width, m_height;

	R3DSDK::Clip *m_clip{ nullptr };
};


#endif

