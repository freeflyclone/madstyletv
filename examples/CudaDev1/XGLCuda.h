#include "ExampleXGL.h"
#include "XGLCudaInterop.h"

extern cudaError_t LaunchKernel(XGLVertexAttributes*, int, int, float);

class XGLCuda : public XGLShape {
public:
	XGLCuda(XGL *px);

	// Run the CUDA kernel defined in <projectName>Kernel.cu
	// This can be (should be?) executed within an AnimationFunction()
	// for this shape. In the Animate() phase, no OpenGL state is being
	// manipulated, so that seems like the ideal place to map OpenGL buffers
	// to CUDA, run the kernel, then unmap OpenGL buffers.
	void RunKernel(float);

	void Draw();

private:
	XGL* pXgl;
	cudaGraphicsResource *cudaVboResource;
	XGLVertexAttributes* dptr;
	size_t numBytes;
};

