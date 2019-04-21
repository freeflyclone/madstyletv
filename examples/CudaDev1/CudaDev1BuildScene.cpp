/**************************************************************
** CudaDev1BuildScene.cpp
**
** CUDA/OpenGL interop example,  from simpleGL CUDA sample.
**
** The CUDA kernel twiddles vertex positions, making a grid
** mesh and modulating the Z position with a 2D sine wave
** function, producing a traveling wave across the mesh.
**
** Rendered as a pixel point cloud.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLCuda.h"

const int meshWidth = 160;
const int meshHeight = 88; // multiple of 8, because of triangle strip rendering

XGLCuda::XGLCuda(XGL *px) : pXgl(px) {
	xprintf("XGLCuda::XGLCuda()\n");

	cudaError_t cudaStatus;
	int y, x;

	// Not positioning vertices in CPU code, but we still need a VBO,
	// so force a VBO with (meshWidth*meshHeight) vertices to be created.
	v.resize(meshWidth*meshHeight);

	xprintf("v.size(): %d\n", v.size());

	// Init all the XGLVertexAttribute members
	for (auto& vrtx : v) {
		vrtx.v = { 0, 0, 0 };
		vrtx.t = { 0, 0 };
		vrtx.n = { 0, 0, 1 };
		vrtx.c = XGLColors::yellow;
	}

	// build a GL_TRIANGLE_STRIP that covers the mesh points
	for (y = 0; y < meshHeight-2; ) {
		for (x = 0; x < meshWidth; x++) {
			idx.push_back(x + ((y + 1) * meshWidth));
			idx.push_back(x + (y*meshWidth));
		}
		idx.push_back((x - 1) + ((y + 1) * meshWidth));
		y++;
		for (x = meshWidth-1; x >= 0; x--) {
			idx.push_back(x + (y * meshWidth));
			idx.push_back(x + ((y + 1) * meshWidth));
		}
		x++;
		idx.push_back((x) + ((y+1) * meshWidth));
		y++;
	}
	for (x = 0; x < meshWidth; x++) {
		idx.push_back(x + ((y + 1) * meshWidth));
		idx.push_back(x + (y*meshWidth));
	}

	// our VBO may not be bound, so bind it now
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	GL_CHECK("glBindBuffer() failed");

	// this loads the actual XGLVertexAttributes into the bound "vbo"
	glBufferData(GL_ARRAY_BUFFER, v.size()*sizeof(XGLVertexAttributes), v.data(), GL_STATIC_DRAW);
	GL_CHECK("glBufferData() failed");

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	// Register VBO with CUDA.
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// our VBO is bound, let's unbind it so it doesn't get polluted. (just in case)
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	GL_CHECK("glBindBuffer() failed");
}

void XGLCuda::RunKernel(float clock) {
	cudaError_t cudaStatus;

	// Step 3: map VBO resource for writing from CUDA
	cudaStatus = cudaGraphicsMapResources(1, &cudaVboResource, 0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsMapResources() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// the CUDA kernel needs a buffer to work on, so map our previously registered VBO as that buffer
	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&dptr, &numBytes, cudaVboResource);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsResourceGetMappedPointer() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// run the CUDA kernel on that buffer
	LaunchKernel(dptr, meshWidth, meshHeight, clock);

	// unmap the VBO, (ostensibly so that OpenGL driver can use it - untested theory)
	cudaStatus = cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
	if (cudaStatus != cudaSuccess) {
		xprintf("cudaGraphicsResourceGetMappedPointer() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}
}

void XGLCuda::Draw() {
	//glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
	//GL_CHECK("glDrawPoints() failed");

	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
}

void ExampleXGL::BuildScene() {
	XGLCuda *shape;
	float scaleFactor = 4.0f;

//	for (int y = -2; y <= 2; y++) {
//		for (int i = -2; i <= 2; i++) {
	for (int y = 0; y < 1; y++) {
		for (int x = 0; x < 1; x++) {
			AddShape("shaders/specular", [&](){ shape = new XGLCuda(this); return shape; });
			shape->attributes.diffuseColor = XGLColors::yellow;
			shape->model = glm::scale(glm::mat4(), glm::vec3(scaleFactor, scaleFactor, scaleFactor)) * glm::translate(glm::mat4(), glm::vec3(x * scaleFactor, y * scaleFactor, 0));

			shape->SetAnimationFunction([shape](float clock) {
				shape->RunKernel(clock / 33.3333f);
			});
		}
	}
	preferredSwapInterval = 1;
}
