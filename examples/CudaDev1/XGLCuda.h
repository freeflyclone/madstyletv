#include "ExampleXGL.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern int cuda_main();
extern cudaError_t LoadKernel(int *dev_a, int *dev_b, int *dev_c, int n);
const int dataSize = 16;

class XGLCuda : public XGLShape {
public:
	XGLCuda(XGL *px);
	void Draw();

	int a[dataSize], b[dataSize], c[dataSize];
private:
	XGL* pXgl;
};

