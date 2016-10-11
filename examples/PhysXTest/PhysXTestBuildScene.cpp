/**************************************************************
** PhysXTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include "mstv-physx.h"

//MstvPhysx myPhysx;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/diffuse", [&](){ shape = new XGLTriangle(); return shape; });
}
