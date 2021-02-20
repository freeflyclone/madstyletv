#include "ExampleXGL.h"
#include "XGLCudaInterop.h"

extern cudaError_t LaunchKernel(XGLRGBA*, int, int, float);
const int defaultCudaTexWidth = 960;
const int defaultCudaTexHeight = 540;


class XGLCuda : public XGLTexQuad {
public:
	XGLCuda::XGLCuda(int width = defaultCudaTexWidth, int height = defaultCudaTexHeight);

	XGLCuda(XGL *, std::string);

	// Run the CUDA kernel defined in <projectName>Kernel.cu
	// This can be (should be?) executed within an AnimationFunction()
	// for this shape. In the Animate() phase, no OpenGL state is being
	// manipulated, so that seems like the ideal place to map OpenGL buffers
	// to CUDA, run the kernel, then unmap OpenGL buffers.
	void RunKernel(float);

	void Draw();

private:
	int m_width, m_height;
	//XGL* pXgl;
	GLuint pbo;
	cudaGraphicsResource *cudaPboResource;
	XGLRGBA* dptr;
	size_t numBytes;
	std::string fileName;
};

