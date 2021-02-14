#pragma once

#include "XGLMemoryPool.h"

class XGLREDCuda : public R3DSDK::REDCuda, public R3DSDK::GpuDecoder
{
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
	void CpuCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus);
	unsigned char* AlignedMalloc(size_t & sizeNeeded);

	R3DSDK::REDCuda * OpenCuda(int & deviceId);

	volatile bool firstFrameDecoded = false;

	void FirstFrameCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus);

	int CUDA_DEVICE_ID = 0;
	volatile int cpuDone = 0;
	volatile int gpuDone = 0;
};

