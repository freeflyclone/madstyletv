/**************************************************************
** Example06BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse, and joystick movement of the triangle
** around the ground plane with the first 2 axes of the first
** joystick in the system.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	AddProportionalFunc("Xbox360Controller0", [this](float v) { xprintf("Value: %0.4f\n", v); });
}
