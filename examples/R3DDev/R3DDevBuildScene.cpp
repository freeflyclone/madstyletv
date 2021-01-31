/**************************************************************
** R3DDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include "R3DSDK.h"
using namespace R3DSDK;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	InitializeStatus initStat;

	initStat = InitializeSdk(".", 0);

	xprintf("initStat: %d\n", initStat);

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
