/**************************************************************
** PhysXTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/simple", [&](){ shape = new XGLTriangle(); return shape; });
}
