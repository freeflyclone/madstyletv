#include "XGLREDCuda.h"
#include "xutils.h"

#define printf xprintf

XGLREDCuda::XGLREDCuda(std::string cn) : clipName(cn) {
	// initialize SDK
	R3DSDK::InitializeStatus init_status = R3DSDK::InitializeSdk(".", OPTION_RED_CUDA);
	if (init_status != R3DSDK::ISInitializeOK)
	{
		R3DSDK::FinalizeSdk();
		printf("Failed to load R3DSDK Lib: %d\n", init_status);
		return;
	}

	m_clip = new R3DSDK::Clip(clipName.c_str());
	if (m_clip->Status() != R3DSDK::LSClipLoaded)
	{
		printf("Failed to load clip %d", m_clip->Status());
		return;
	}

	m_width = m_clip->Width();
	m_height = m_clip->Height();

	// initialize a R3DSDK::REDCuda nugget - does debayering on the GPU
	m_pREDCuda = OpenCuda(CUDA_DEVICE_ID);

	// initialize a R3DSDK::GpuDecoder nugget - does decoding (wavelet decompression) on a frame
	m_pGpuDecoder = new R3DSDK::GpuDecoder();
	m_pGpuDecoder->Open();

	GenR3DInterleavedTextureBuffer(m_width, m_height);
	AllocatePBOOutputBuffer();

	// there doesn't appear to be any harm in spawning these here.
	std::thread* gpuThread = new std::thread(std::bind(&XGLREDCuda::GpuThread, this, 0));
	std::thread* completionThread = new std::thread(std::bind(&XGLREDCuda::CompletionThread, this));
}

void XGLREDCuda::StartVideoDecode(int frame) {
	// allocate an AsyncDecompressJob, and fill it in.
	R3DSDK::AsyncDecompressJob* job = new R3DSDK::AsyncDecompressJob();
	job->Clip = m_clip;
	job->Mode = R3DSDK::DECODE_FULL_RES_PREMIUM;
	job->OutputBufferSize = R3DSDK::GpuDecoder::GetSizeBufferNeeded(*job);
	size_t adjustedSize = job->OutputBufferSize;

	// only one output buffer, since it doesn't get free()'d anywhere (and we don't really need it?) 
	if (m_outputBuffer == nullptr)
		m_outputBuffer = AlignedMalloc(adjustedSize);

	job->OutputBuffer = m_outputBuffer;
	job->VideoFrameNo = frame;
	job->VideoTrackNo = 0;
	job->Callback = CpuCallback;
	job->PrivateData = this;

	// Okay... we should be ready to go.
	if (m_pGpuDecoder->DecodeForGpuSdk(*job) != R3DSDK::DSDecodeOK)
	{
		printf("GPU decode submit failed\n");
		return;
	}
}

void XGLREDCuda::GenR3DInterleavedTextureBuffer(const int width, const int height) {
	GLuint texId;

	glGenTextures(1, &texId);
	glActiveTexture(GL_TEXTURE0 + numTextures);
	glBindTexture(GL_TEXTURE_2D, texId);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT, (void *)(nullptr));

	GL_CHECK("Eh, something failed");

	AddTexture(texId);
}

void XGLREDCuda::AllocatePBOOutputBuffer() {
	cudaError_t cudaStatus;

	// generate a Pixel Buffer Object for CUDA to write to.
	glGenBuffers(1, &pbo);
	GL_CHECK("glGenBuffers() didn't work");

	// bind it to the pixel unpack buffer (texture source)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	GL_CHECK("glBindBuffer() didn't work");

	// tell OpenGL to allocate the pixel buffer, (without giving it any initial data)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, defaultCudaTexWidth * defaultCudaTexHeight * sizeof(XGLREDCudaRGB16), NULL, GL_DYNAMIC_DRAW);
	GL_CHECK("glBufferData() didn't work");

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	// Register PBO with CUDA.  We can now pass this buffer to the CUDA kernel, see RunKernel() below.
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// map PBO resource for writing from CUDA
	cudaStatus = cudaGraphicsMapResources(1, &cudaPboResource, 0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsMapResources() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// the R3DSDK::DebayerCudaJob needs an output buffer.
	// Rather than allocate and deallocate on the fly (as the example code does), let's just get
	// a mapped (for CUDA) ptr to our previously registered PBO.
	//
	// I believe that leaving this persistently mapped is OK.  Certainly works for the 1st frame.
	// Actual successive decodings may prove me wrong.
	//
	// The GpuThread this runs in balks at this call because the graphics context isn't current.
	// This persistent mapping is intended to circumvent the need for setting the context in the GpuThread().
	// Although that *might* be doable.  So map it here, hold onto the device memory pointer and size as
	// state in this object, so GpuThread can just use it.
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&pPboCudaMemory, &pboCudaMemorySize, cudaPboResource);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsResourceGetMappedPointer() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	return;
}

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
	R3DSDK::DebayerCudaJob *data = m_pREDCuda->createDebayerJob();

	m_width = job->Clip->Width();
	m_height = job->Clip->Height();

	data->raw_host_mem = job->OutputBuffer;
	data->mode = job->Mode;
	data->imageProcessingSettings = imageProcessingSettings;
	data->pixelType = pixelType;

	//create raw buffer on the Cuda device
	cudaError_t err = XGLCudaMemoryPool::cudaMalloc(&(data->raw_device_mem), job->OutputBufferSize);

	if (err != cudaSuccess)
	{
		printf("Failed to allocate raw frame on GPU: %d\n", err);
		m_pREDCuda->releaseDebayerJob(data);
		return NULL;
	}

	// specify output buffer as our pre-mapped PBO buffer.
	data->output_device_mem = pPboCudaMemory;
	data->output_device_mem_size = pboCudaMemorySize;

	if (err != cudaSuccess)
	{
		printf("Failed to allocate result frame on card %d\n", err);
		XGLCudaMemoryPool::cudaFree(data->raw_device_mem);
		m_pREDCuda->releaseDebayerJob(data);
		return NULL;
	}

	return data;
}

void  XGLREDCuda::DebayerFree(R3DSDK::DebayerCudaJob * job)
{
	XGLCudaMemoryPool::cudaFree(job->raw_device_mem);
	//XGLCudaMemoryPool::cudaFree(job->output_device_mem);
	m_pREDCuda->releaseDebayerJob(job);
}

void XGLREDCuda::CompletionThread()
{
	//printf("Inside CompletionThread()\n");
	for (;;)
	{
		R3DSDK::AsyncDecompressJob * job = NULL;

		CompletionQueue.pop(job);

		// exit thread
		if (job == NULL)
			break;

		XGLREDCuda* pRedCuda = reinterpret_cast<XGLREDCuda*>(job->PrivateData);
		R3DSDK::DebayerCudaJob* cudaJob = pRedCuda->m_debayerJob;

		cudaJob->completeAsync();

		// frame ready for use or download etc.
		//("Completed frame %d.\n", gpuDone);

		for (auto cf : completionFuncs)
			cf(pRedCuda);

		gpuDone++;

		DebayerFree(cudaJob);
/*
		job->PrivateData = NULL;

		// queue up next frame for decode
		if (cpuDone < TOTAL_FRAMES)
		{
			cpuDone++;
			if (m_pGpuDecoder->DecodeForGpuSdk(*job) != R3DSDK::DSDecodeOK)
			{
				printf("CPU decode submit failed\n");
			}
		}
*/
	}
}

void XGLREDCuda::GpuThread(int device)
{
	printf("Inside GpuThread()!\n");

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
		XGLREDCuda* pRedCuda = (XGLREDCuda*)job->PrivateData;

		// exit thread
		if (job == NULL)
			break;

		R3DSDK::ImageProcessingSettings * ips = new R3DSDK::ImageProcessingSettings();
		job->Clip->GetDefaultImageProcessingSettings(*ips);

		m_debayerJob = DebayerAllocate(job, ips, R3DSDK::PixelType_16Bit_RGB_Interleaved);

		if (err != cudaSuccess)
		{
			printf("Failed to move raw frame to card %d\n", err);
		}

		int idx = frameCount++ % NUM_STREAMS;

		R3DSDK::REDCuda::Status status = m_pREDCuda->processAsync(device, stream[idx], m_debayerJob, err);

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
			job->PrivateData = pRedCuda;
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
	XGLREDCuda* pThis = (XGLREDCuda*)item->PrivateData;
	pThis->JobQueue.push(item);
}

unsigned char * XGLREDCuda::AlignedMalloc(size_t & sizeNeeded)
{
	// alloc 15 bytes more to make sure we can align the buffer in case it isn't
	unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 15U);;

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

void XGLREDCuda::AddCompletionFunction(XGLREDCuda::CompletionFunc fn) {
	completionFuncs.push_back(fn);
}

void XGLREDCuda::Draw() {
	glEnable(GL_BLEND);
	GL_CHECK("glEnable(GL_BLEND) failed");

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GL_CHECK("glBlendFunc() failed");

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	GL_CHECK("glBindBuffer() failed");

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_SHORT, (GLvoid *)0);
	GL_CHECK("glTexSubImage2D() failed");

	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	GL_CHECK("glBindBuffer() failed");

	glDisable(GL_BLEND);
	GL_CHECK("glDisable(GL_BLEND) failed");
}
