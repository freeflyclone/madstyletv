/**************************************************************
** ComputeShaderTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

XGLShader *computeShader;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	computeShader = new XGLShader("shaders/compute-shader");
	computeShader->CompileCompute(pathToAssets + "/shaders/compute-shader");

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
