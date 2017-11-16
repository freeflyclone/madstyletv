/**************************************************************
** GravityBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include "xphybody.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/specular", [&](){ shape = new XGLSphere(0.5f, 64); return shape; });
	shape->p = { 0.0f, 0.0f, 10.0f };
}
