/**
* File: XGLCudaInterop.h
*
* Description:
*	XGL.h includes JSON.h which redefines something defined in cuda_runtime.h
*   This header allows for including the portions of XGL that XGLCuda needs
*   for interoperability with CUDA.
*/
#ifndef XGLCUDAINTEROP_H
#define XGLCUDAINTEROP_H

#include <vector>
#include <math.h>
#include <stdint.h>

#include "glew.h"
#include "wglew.h"
#include "glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "matrix_transform.hpp"
#include "type_ptr.hpp"

#include "xglprimitives.h"
#include "xutils.h"

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

struct XGLRGBA {
	uint8_t r, g, b, a;
};


#endif // XGLCUDAINTEROP_H