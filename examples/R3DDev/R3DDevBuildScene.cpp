/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"
#include <cstdlib>
#include <cuda_runtime.h>

#include "R3DSDK.h"
#include "R3DSDKCuda.h"
#include <R3DSDKDefinitions.h>

using namespace R3DSDK;

volatile bool decodeDone = false;
void asyncCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
{
	printf("Frame callback: %d\n", decodeStatus);
	*((R3DSDK::DecodeStatus*)item->PrivateData) = decodeStatus;
	decodeDone = true;
}

unsigned char * AlignedMalloc(size_t & sizeNeeded)
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

R3DSDK::REDCuda::Status Debayer(void *source_raw_host_memory_buffer, size_t raw_buffer_size, R3DSDK::VideoPixelType pixelType, R3DSDK::VideoDecodeMode mode, R3DSDK::ImageProcessingSettings &ips, void **result_host_memory_buffer, size_t &result_buffer_size)
{
	//setup Cuda for the current thread
	int deviceId = 0;
	cudaDeviceProp deviceProp;
	cudaError_t err = cudaChooseDevice(&deviceId, &deviceProp);
	if (err != cudaSuccess)
	{
		printf("Failed to move raw frame to card %d\n", err);
		return R3DSDK::REDCuda::Status_UnableToUseGPUDevice;
	}

	err = cudaSetDevice(deviceId);
	if (err != cudaSuccess)
	{
		printf("Failed to move raw frame to card %d\n", err);
		return R3DSDK::REDCuda::Status_UnableToUseGPUDevice;
	}

	cudaStream_t stream;

	err = cudaStreamCreate(&stream);
	if (err != cudaSuccess)
	{
		printf("Failed to create stream %d\n", err);
		return R3DSDK::REDCuda::Status_UnableToUseGPUDevice;
	}

	//SETUP YOUR CUDA API FUNCTION POINTERS
	//This is left for the SDK user to do, just incase you wish to use your own custom routines in place of cuda calls
	R3DSDK::EXT_CUDA_API api;
	api.cudaFree = ::cudaFree;
	api.cudaFreeArray = ::cudaFreeArray;
	api.cudaFreeHost = ::cudaFreeHost;
	api.cudaFreeMipmappedArray = ::cudaFreeMipmappedArray;
	api.cudaHostAlloc = ::cudaHostAlloc;
	api.cudaMalloc = ::cudaMalloc;
	api.cudaMalloc3D = ::cudaMalloc3D;
	api.cudaMalloc3DArray = ::cudaMalloc3DArray;
	api.cudaMallocArray = ::cudaMallocArray;
	api.cudaMallocHost = ::cudaMallocHost;
	api.cudaMallocMipmappedArray = ::cudaMallocMipmappedArray;
	api.cudaMallocPitch = ::cudaMallocPitch;


	//create the REDCuda class
	R3DSDK::REDCuda *redcuda = new R3DSDK::REDCuda(api);
	R3DSDK::REDCuda::Status status = redcuda->checkCompatibility(deviceId, stream, err);
	if (R3DSDK::REDCuda::Status_Ok != status)
	{

		if (status == R3DSDK::REDCuda::Status_UnableToLoadLibrary)
		{
			printf("Error: Unable to load the REDCuda dynamic library %d, This could be caused by the file being missing, or potentially missing the cudart dynamic library.\n", status);
			return status;
		}

		printf("Compatibility Check Failed\n");
		return R3DSDK::REDCuda::Status_UnableToUseGPUDevice;
	}

	//allocate the debayer job
	R3DSDK::DebayerCudaJob *data = redcuda->createDebayerJob();

	data->imageProcessingSettings = new R3DSDK::ImageProcessingSettings();
	data->mode = mode; //Quality mode

	data->raw_host_mem = source_raw_host_memory_buffer;
	memcpy(data->imageProcessingSettings, &ips, sizeof(R3DSDK::ImageProcessingSettings));//Image Processing Settings to apply
	data->pixelType = pixelType;
	//verify buffer size
	if (raw_buffer_size == 0)
	{
		return R3DSDK::REDCuda::Status_InvalidJobParameter_raw_host_mem;
	}

	//create raw buffer on the Cuda device
	err = cudaMalloc(&(data->raw_device_mem), raw_buffer_size);


	//upload the result from the Async Decoder DecodeForGpuSdk to Device
	err = cudaMemcpy(data->raw_device_mem, data->raw_host_mem, raw_buffer_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Failed to move raw frame to card %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	//wait for completion in the worst possible way - real apps should avoid cudaDeviceSynchronize
	cudaDeviceSynchronize();

	//setup result pointer - will be a device memory pointer
	result_buffer_size = R3DSDK::DebayerCudaJob::ResultFrameSize(data);

	//YOU MUST specify an existing buffer for the result image
	//Set DebayerCudaJob::output_device_mem_size >= result_buffer_size
	//and a pointer to the device buffer in DebayerCudaJob::output_device_mem
	err = cudaMalloc(&(data->output_device_mem), result_buffer_size);
	data->output_device_mem_size = result_buffer_size;
	if (err != cudaSuccess)
	{
		printf("Failed to allocate result frame on card %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	//verbose clearing of buffer - isolates this test run from the previous.
	err = cudaMemset(data->output_device_mem, 0, result_buffer_size);
	if (err != cudaSuccess)
	{
		printf("Failed to clear result frame prior to use on card %d\n", err);
	}


	bool process_debayer_async = true;
	//Run the debayer on the given buffers.
	if (process_debayer_async)
	{
		R3DSDK::REDCuda::Status status = redcuda->processAsync(deviceId, stream, data, err);

		if (status != R3DSDK::REDCuda::Status_Ok)
		{
			printf("Failed to process frame %d", status);
			if (err != cudaSuccess)
			{
				printf(" Cuda Error: %d\n", err);
				return status;
			}
			printf("\n");
			return status;
		}



		//enqueue other cuda comamnds etc here to the stream.
		//you can do any cuda stream synchronization here you need.

		//This will ensure all objects used for the frame are disposed of.
		//This call will block until the debayer on the stream executes, 
		// if the debayer has already executed no block will occur
		data->completeAsync();
	}
	else
	{
		//Process simply condenses the processAsync call with completeAsync, effectively blocking until all SDK tasks for this frame have completed.
		//The downside to using process insead of processAsync is that you are stuck synchronizing before executing your own kernels (which if on the same queue do not need synchronization).
		//The benefit is that it simplifies the usage of the SDK.
		R3DSDK::REDCuda::Status status = redcuda->process(deviceId, stream, data, err);

		if (status != R3DSDK::REDCuda::Status_Ok)
		{
			printf("Failed to process frame %d", status);
			if (err != cudaSuccess)
			{
				printf(" Cuda Error: %d\n", err);
				return status;
			}
			printf("\n");
			return status;
		}
	}

	//Optional: Copy the image back to cpu/host memory to be written to disk later

	//allocate the result buffer in host memory.
	if (result_buffer_size != data->output_device_mem_size)
	{
		printf("Result Buffer size does not match expected size: Expected: %lu Actual: %lu\n", result_buffer_size, data->output_device_mem_size);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}
	*result_host_memory_buffer = malloc(result_buffer_size);//RGB 16 Interleaved

	//read the GPU buffer back to the host memory result buffer. - Note this is not always the optimal way to read back. (Use pinned memory in a real app)
	err = cudaMemcpy(*result_host_memory_buffer, data->output_device_mem, result_buffer_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("Failed to read result frame from card %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	//ensure the read is complete.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		printf("Failed to finish after reading result frame from card %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}
	//final image data is now in result_host_memory_buffer

	//Tear down Cuda
	//free memory objects
	err = cudaFree(data->output_device_mem);
	if (err != cudaSuccess)
	{
		printf("Failed to release memory object %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	err = cudaFree(data->raw_device_mem);
	if (err != cudaSuccess)
	{
		printf("Failed to release memory object %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	delete data->imageProcessingSettings;
	data->imageProcessingSettings = NULL;

	redcuda->releaseDebayerJob(data);
	//tear down redCuda
	delete redcuda;

	//release other cuda objects.
	err = cudaStreamDestroy(stream);
	if (err != cudaSuccess)
	{
		printf("Failed to release stream %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}

	err = cudaDeviceReset();
	if (err != cudaSuccess)
	{
		printf("Failed reset the device %d\n", err);
		return R3DSDK::REDCuda::Status_ErrorProcessing;
	}
	return R3DSDK::REDCuda::Status_Ok;
}//end Debayer

R3DSDK::DecodeStatus Decompress(const char *filename, size_t frame_number, R3DSDK::ImageProcessingSettings &ips_to_be_filled_with_defaults, R3DSDK::VideoDecodeMode mode, void **raw_buffer, size_t &raw_buffer_size, size_t &raw_buffer_aligned_ptr_adjustment)
{
	R3DSDK::Clip *clip = new R3DSDK::Clip(filename);
	if (clip->Status() != R3DSDK::LSClipLoaded)
	{
		printf("Failed to load clip %d\n", clip->Status());
		return R3DSDK::DSNoClipOpen;
	}

	//setup R3DSDK to decode a frame
	R3DSDK::AsyncDecompressJob *job = new R3DSDK::AsyncDecompressJob();
	job->Clip = clip;
	job->Mode = mode;

	raw_buffer_size = R3DSDK::AsyncDecoder::GetSizeBufferNeeded(*job);
	raw_buffer_aligned_ptr_adjustment = raw_buffer_size;
	*raw_buffer = AlignedMalloc(raw_buffer_aligned_ptr_adjustment);
	
	job->OutputBuffer = *raw_buffer;
	job->OutputBufferSize = raw_buffer_size;
	job->PrivateData = new R3DSDK::DecodeStatus();
	*((R3DSDK::DecodeStatus*)job->PrivateData) = R3DSDK::DSOutputBufferInvalid;
	job->VideoFrameNo = 0;
	job->VideoTrackNo = 0;
	job->Callback = asyncCallback;

	R3DSDK::AsyncDecoder *r3dAsync = new R3DSDK::AsyncDecoder();
	r3dAsync->Open(r3dAsync->ThreadsAvailable());

	// ----------------------- Loop this for video -------------------------------
	//Decode a frame using the R3DSDK
	R3DSDK::DecodeStatus decompress_status = r3dAsync->DecodeForGpuSdk(*job);
	if (decompress_status != R3DSDK::DSDecodeOK)
	{
		printf("Failed to start decompression %d\n", decompress_status);

		//abort clearing up allocated memory.
		r3dAsync->Close();
		delete (R3DSDK::DecodeStatus*)job->PrivateData;
		delete job;
		delete r3dAsync;
		clip->Close();
		delete clip;
		// free the original pointer, not the one adjusted for alignment
		free(((unsigned char *)*raw_buffer) - raw_buffer_aligned_ptr_adjustment);
		return decompress_status;
	}
	// ---------------------------End loop this for video -----------------------

	// I added the thread sleep.
	while (!decodeDone)
	{
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
	}

	R3DSDK::DecodeStatus callback_status = *((R3DSDK::DecodeStatus*)job->PrivateData);
	//tear down the Async decoder as we are done with that now
	r3dAsync->Close();
	delete (R3DSDK::DecodeStatus*)job->PrivateData;
	delete job;
	delete r3dAsync;

	//get the image processing settings for the frame from the clip.

	clip->GetDefaultImageProcessingSettings(ips_to_be_filled_with_defaults);

	//we no longer need the clip around.
	clip->Close();
	delete clip;

	return callback_status;
}//end Decompress

//------------------------------------------ Start of R3DPlayer ----------------------------------------
class R3DPlayer : public XGLTexQuad {
public:
	R3DPlayer(const std::string& fname) : XGLTexQuad() {
		Clip* clip = InitializeClip(fname);

		m_width = clip->Width();
		m_height = clip->Height();

		AllocateAlignedHostBuffer(clip);

		//CPUDecode1stFrame(clip);
		CudaDecode1stFrame(clip);

		//GenR3DPlanarTextureBuffer(m_width, m_height);
		GenR3DInterleavedTextureBuffer(m_width, m_height);

		delete clip;

		FinalizeSdk();
	};

	~R3DPlayer() {
		if (m_unalignedImgbuffer)
			delete m_unalignedImgbuffer;
	}

	Clip* InitializeClip(const std::string& fname) {
		fileName = fname;
		const char* clipName = fileName.c_str();

		InitializeStatus initStat;

		initStat = InitializeSdk(".", OPTION_RED_CUDA);
		if (initStat != ISInitializeOK)
		{
			xprintf("Failed to initialize SDK: %d\n", initStat);
			return nullptr;
		}

		// load the clip
		Clip *clip = new Clip(clipName);

		// let the user know if this failed
		if (clip->Status() != LSClipLoaded)
		{
			xprintf("Error loading %s: %d\n", clipName, clip->Status());
			delete clip;
			FinalizeSdk();
			return nullptr;
		}

		xprintf("Loaded %s\n", clipName);
		return clip;
	}

	uint16_t* AllocateAlignedHostBuffer(Clip* clip) {
		size_t width = clip->Width();
		size_t height = clip->Height();

		// three channels (RGB) in 16-bit (2 bytes) requires this much memory:
		m_imgSize = width * height * 3 * 2;
		size_t adjusted = m_imgSize + 16;

		// alloc this memory 16-byte aligned
		m_unalignedImgbuffer = new uint8_t[adjusted];
		if (m_unalignedImgbuffer == NULL) {
			xprintf("Failed to allocate %d bytes of memory for output image\n", static_cast<unsigned int>(m_imgSize));
			return nullptr;
		}

		m_imgbuffer = (uint16_t*)(std::align(16, m_imgSize, (void*&)m_unalignedImgbuffer, adjusted));
		return m_imgbuffer;
	}

	void CPUDecode1stFrame(Clip* clip) {
		VideoDecodeJob job;

		// setup decoder parameters
		job.BytesPerRow = m_width * 2U;
		job.OutputBufferSize = m_imgSize;
		job.Mode = DECODE_FULL_RES_PREMIUM;
		job.OutputBuffer = m_imgbuffer;
		job.PixelType = PixelType_16Bit_RGB_Planar;

		// decode the first frame (0) of the clip
		xprintf("Image is %d x %d\n", m_width, m_height);

		if (clip->DecodeVideoFrame(0U, job) != DSDecodeOK)
		{
			xprintf("Decode failed?\n");
			delete clip;
			FinalizeSdk();
			return;
		}
	}

	int CudaDecode1stFrame(Clip* clip) {
		xprintf("Inside: %s()\n", __FUNCTION__);
		R3DSDK::VideoDecodeMode mode = R3DSDK::DECODE_FULL_RES_PREMIUM;
		R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;
		R3DSDK::ImageProcessingSettings *ips = new R3DSDK::ImageProcessingSettings();
		void *raw_buffer = NULL;
		size_t raw_buffer_size = 0;
		size_t raw_buffer_aligned_ptr_adjustment = 0;

		//Decompress the R3D Source frame into a Raw Buffer using the AsnycDecompressor
		R3DSDK::DecodeStatus decompress_status = Decompress(fileName.c_str(), 0, *ips, mode, &raw_buffer, raw_buffer_size, raw_buffer_aligned_ptr_adjustment);
		if (decompress_status != R3DSDK::DSDecodeOK)
		{
			//failed to decode
			printf("Error decompressing frame: %d\n", decompress_status);
			R3DSDK::FinalizeSdk();
			return 5;
		}

		//size the result buffer will be returned from debayer
		size_t result_buffer_size = 0;
		//buffer in host memory filled by example code in debayer routine.
		void *result_host_memory_buffer = NULL;

		//Debayer Logic has been completely seperated to illustrate it's independence from the decompression step.
		//Debayer a Raw Buffer into a frame of the selected pixel type
		R3DSDK::REDCuda::Status debayer_status = Debayer(raw_buffer, raw_buffer_size, pixelType, mode, *ips, &result_host_memory_buffer, result_buffer_size);
		if (raw_buffer)
		{
			// free the original pointer, not the one adjusted for alignment
			free(((unsigned char *)raw_buffer) - raw_buffer_aligned_ptr_adjustment);
		}

		// wrong format, but there's an image there now.
		memcpy(m_imgbuffer, result_host_memory_buffer, result_buffer_size);

		return 0;
	}

	void GenR3DPlanarTextureBuffer(const int width, const int height) {
		// Output of R3D decoder is 16 bit planar, in RGB order
		// Each plane gets it's own texture unit, cuz that's super
		// easy, at the expense of being sub-optimal
		for (int i = 0; i < 3; i++) {
			GLuint texId;

			glGenTextures(1, &texId);
			glActiveTexture(GL_TEXTURE0 + numTextures);
			glBindTexture(GL_TEXTURE_2D, texId);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, (void *)(m_imgbuffer + (width*height*i)));

			GL_CHECK("Eh, something failed");

			AddTexture(texId);
		}
	}

	void GenR3DInterleavedTextureBuffer(const int width, const int height) {
		GLuint texId;

		glGenTextures(1, &texId);
		glActiveTexture(GL_TEXTURE0 + numTextures);
		glBindTexture(GL_TEXTURE_2D, texId);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT, (void *)(m_imgbuffer));

		GL_CHECK("Eh, something failed");

		AddTexture(texId);
	}

private:
	int m_width{ 0 };
	int m_height{ 0 };
	uint8_t *m_unalignedImgbuffer{ nullptr };
	uint16_t *m_imgbuffer{ nullptr };
	size_t m_imgSize;
	std::string fileName;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex", [&](){ shape = new R3DPlayer(r3DClipName); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
