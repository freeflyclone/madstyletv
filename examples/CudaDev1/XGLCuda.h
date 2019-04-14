#include "ExampleXGL.h"
#include "XGLCudaInterop.h"

extern cudaError_t LaunchKernel(XGLVertexAttributes*, int, int, float);

class XGLCuda : public XGLShape {
public:
	XGLCuda(XGL *px);
	void RunKernel(float);
	void Draw();

private:
	XGL* pXgl;
	cudaGraphicsResource *cudaVboResource;
	XGLVertexAttributes* dptr;
	size_t numBytes;
};

