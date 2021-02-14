#include "XGLREDCuda.h"

void XGLREDCuda::getCurrentTimestamp()
{
	time_t tt;
	time(&tt);
	tm * timeinfo = localtime(&tt);

	using std::chrono::system_clock;
	auto currentTime = std::chrono::system_clock::now();

	auto transformed = currentTime.time_since_epoch().count() / 1000000;
	auto millis = transformed % 1000;

	char buffer[80];
	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M.%S", timeinfo);

	sprintf(buffer, "%s:%03d", buffer, (int)millis);
	printf("Time is %s ", buffer);
}

R3DSDK::DebayerCudaJob* XGLREDCuda::DebayerAllocate(const R3DSDK::AsyncDecompressJob * job, R3DSDK::ImageProcessingSettings * imageProcessingSettings, R3DSDK::VideoPixelType pixelType)
{
	//allocate the debayer job
	R3DSDK::DebayerCudaJob *data = createDebayerJob();

	data->raw_host_mem = job->OutputBuffer;
	data->mode = job->Mode;
	data->imageProcessingSettings = imageProcessingSettings;
	data->pixelType = pixelType;

	//create raw buffer on the Cuda device
	cudaError_t err = XGLCudaMemoryPool::cudaMalloc(&(data->raw_device_mem), job->OutputBufferSize);

	if (err != cudaSuccess)
	{
		printf("Failed to allocate raw frame on GPU: %d\n", err);
		releaseDebayerJob(data);
		return NULL;
	}

	data->output_device_mem_size = R3DSDK::DebayerCudaJob::ResultFrameSize(data);

	//YOU MUST specify an existing buffer for the result image
	//Set DebayerCudaJob::output_device_mem_size >= result_buffer_size
	//and a pointer to the device buffer in DebayerCudaJob::output_device_mem
	err = XGLCudaMemoryPool::cudaMalloc(&(data->output_device_mem), data->output_device_mem_size);

	if (err != cudaSuccess)
	{
		printf("Failed to allocate result frame on card %d\n", err);
		XGLCudaMemoryPool::cudaFree(data->raw_device_mem);
		releaseDebayerJob(data);
		return NULL;
	}

	return data;
}

void  XGLREDCuda::DebayerFree(R3DSDK::DebayerCudaJob * job)
{
	XGLCudaMemoryPool::cudaFree(job->raw_device_mem);
	XGLCudaMemoryPool::cudaFree(job->output_device_mem);
	releaseDebayerJob(job);
}

void XGLREDCuda::CompletionThread()
{
	for (;;)
	{
		R3DSDK::AsyncDecompressJob * job = NULL;

		CompletionQueue.pop(job);

		// exit thread
		if (job == NULL)
			break;

		R3DSDK::DebayerCudaJob * cudaJob = reinterpret_cast<R3DSDK::DebayerCudaJob *>(job->PrivateData);

		cudaJob->completeAsync();

		// frame ready for use or download etc.
		printf("Completed frame %d .\n", gpuDone);

		gpuDone++;

		DebayerFree(cudaJob);

		job->PrivateData = NULL;

		// queue up next frame for decode
		if (cpuDone < TOTAL_FRAMES)
		{
			cpuDone++;
			if (DecodeForGpuSdk(*job) != R3DSDK::DSDecodeOK)
			{
				printf("CPU decode submit failed\n");
			}
		}
	}
}

void XGLREDCuda::GpuThread(int device)
{
	cudaSetDevice(device);

	cudaStream_t stream[NUM_STREAMS];

	cudaError_t err;

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		err = cudaStreamCreate(&stream[i]);
	}

	if (err != cudaSuccess)
	{
		printf("Failed to create stream %d\n", err);
		return;
	}

	int frameCount = 0;

	while (true)
	{
		R3DSDK::AsyncDecompressJob * job = NULL;

		JobQueue.pop(job);

		// exit thread
		if (job == NULL)
			break;

		const R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;

		R3DSDK::ImageProcessingSettings * ips = new R3DSDK::ImageProcessingSettings();
		job->Clip->GetDefaultImageProcessingSettings(*ips);

		R3DSDK::DebayerCudaJob * cudaJob = DebayerAllocate(job, ips, pixelType);

		if (err != cudaSuccess)
		{
			printf("Failed to move raw frame to card %d\n", err);
		}

		int idx = frameCount++ % NUM_STREAMS;

		R3DSDK::REDCuda::Status status = processAsync(device, stream[idx], cudaJob, err);

		if (status != R3DSDK::REDCuda::Status_Ok)
		{
			printf("Failed to process frame, error %d.", status);

			if (err != cudaSuccess)
				printf(" Cuda Error: %d\n", err);
			else
				printf("\n");
		}
		else
		{
			job->PrivateData = cudaJob;
			CompletionQueue.push(job);
		}
	}

	// cleanup
	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
}

void XGLREDCuda::CpuCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
{
	JobQueue.push(item);
}

unsigned char * XGLREDCuda::AlignedMalloc(size_t & sizeNeeded)
{
	// alloc 15 bytes more to make sure we can align the buffer in case it isn't
	unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 15U);

	if (!buffer)
		return NULL;

	sizeNeeded = 0U;

	// cast to a 32-bit or 64-bit (depending on platform) integer so we can do the math
	uintptr_t ptr = (uintptr_t)buffer;

	// check if it's already aligned, if it is we're done
	if ((ptr % 16U) == 0U)
		return buffer;

	// calculate how many bytes we need
	sizeNeeded = 16U - (ptr % 16U);

	return buffer + sizeNeeded;
}

R3DSDK::REDCuda * XGLREDCuda::OpenCuda(int & deviceId)
{
	//setup Cuda for the current thread
	cudaDeviceProp deviceProp;
	cudaError_t err = cudaChooseDevice(&deviceId, &deviceProp);
	if (err != cudaSuccess)
	{
		printf("Failed to move raw frame to card %d\n", err);
		return NULL;
	}

	err = cudaSetDevice(deviceId);
	if (err != cudaSuccess)
	{
		printf("Failed to move raw frame to card %d\n", err);
		return NULL;
	}

	//SETUP YOUR CUDA API FUNCTION POINTERS
	R3DSDK::EXT_CUDA_API api;
	api.cudaFree = XGLCudaMemoryPool::cudaFree;
	api.cudaFreeArray = XGLCudaMemoryPool::cudaFreeArray;
	api.cudaFreeHost = XGLCudaMemoryPool::cudaFreeHost;
	api.cudaFreeMipmappedArray = ::cudaFreeMipmappedArray;
	api.cudaHostAlloc = XGLCudaMemoryPool::cudaHostAlloc;
	api.cudaMalloc = XGLCudaMemoryPool::cudaMalloc;
	api.cudaMalloc3D = ::cudaMalloc3D;
	api.cudaMalloc3DArray = XGLCudaMemoryPool::cudaMalloc3DArray;
	api.cudaMallocArray = XGLCudaMemoryPool::cudaMallocArray;
	api.cudaMallocHost = XGLCudaMemoryPool::cudaMallocHost;
	api.cudaMallocMipmappedArray = ::cudaMallocMipmappedArray;
	api.cudaMallocPitch = ::cudaMallocPitch;

	//CREATE THE REDCuda CLASS
	return new R3DSDK::REDCuda(api);
}

void XGLREDCuda::FirstFrameCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
{
	firstFrameDecoded = true;
}
