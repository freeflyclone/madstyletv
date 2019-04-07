/**************************************************************
** CudaDevNvBuildScene.cpp
**
** Develop a CUDA demo, as simple as possible.  This one
** originated as an NVIDIA Cuda template project, which I 
** then massaged into an XGL project by hand. "cuda_main()"
** is in kernel.cu, which gets built with nvcc from CUDA
** toolkit.
**************************************************************/
#include "ExampleXGL.h"

extern int cuda_main();

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	cuda_main();
}
