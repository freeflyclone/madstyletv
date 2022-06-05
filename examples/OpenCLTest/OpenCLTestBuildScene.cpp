/**************************************************************
** OpenCLTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include <vector>
#include "CL/cl.hpp"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty()) {
		xprintf("Doh! no OpenCL platform!\n");
	}

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
