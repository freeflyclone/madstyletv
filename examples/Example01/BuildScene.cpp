/**************************************************************
** For Example01: just demonstrate instantiation of a "ground"
** plane and a single spere, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/simple", [&](){ shape = new XGLSphere(1.0f, 32); return shape; });
}