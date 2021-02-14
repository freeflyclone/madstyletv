/**************************************************************
** CudaTextureBuildScene.cpp
**
** CUDA/OpenGL example, derived from simpleGL CUDA sample.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLCuda.h"

const int width = 960;
const int height = 540;

XGLCuda::XGLCuda(XGL *px, std::string fn) : XGLTexQuad(fn), pXgl(px) {
	xprintf("XGLCuda::XGLCuda()\n");
	cudaError_t cudaStatus;

	// generate a Pixel Buffer Object for CUDA to write to.
	glGenBuffers(1, &pbo);
	GL_CHECK("glGenBuffers() didn't work");

	// bind it to the pixel unpack buffer (texture source)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	GL_CHECK("glBindBuffer() didn't work");

	// tell OpenGL to allocate the pixel buffer, (without giving it any initial data)
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(XGLRGBA), NULL, GL_DYNAMIC_DRAW);
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
}

// runs every render loop iteration
void XGLCuda::RunKernel(float clock) {
	cudaError_t cudaStatus;

	// map PBO resource for writing from CUDA
	cudaStatus = cudaGraphicsMapResources(1, &cudaPboResource, 0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsMapResources() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// the kernel needs a buffer, so get a mapped (for CUDA) ptr to our previously registered PBO
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&dptr, &numBytes, cudaPboResource);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsResourceGetMappedPointer() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// run the CUDA kernel on the mapped buffer
	LaunchKernel(dptr, width, height, clock);

	// unmap the PBO, so that OpenGL driver can use it for rendering.
	cudaStatus = cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsResourceGetMappedPointer() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}
}

void XGLCuda::Draw() {
	glEnable(GL_BLEND);
	GL_CHECK("glEnable(GL_BLEND) failed");

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	GL_CHECK("glBlendFunc() failed");

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	GL_CHECK("glBindBuffer() failed");

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid *)0);
	GL_CHECK("glTexSubImage2D() failed");

	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	GL_CHECK("glBindBuffer() failed");

	glDisable(GL_BLEND);
	GL_CHECK("glDisable(GL_BLEND) failed");
}

void ExampleXGL::BuildScene() {
	XGLCuda *shape;
	float scaleFactor = 4.0f;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/tex", [&shape, imgPath, this](){ shape = new XGLCuda(this, imgPath); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	shape->SetAnimationFunction([shape](float clock) {
		shape->RunKernel(clock / 50.000f);
	});

	preferredSwapInterval = 1;
}
