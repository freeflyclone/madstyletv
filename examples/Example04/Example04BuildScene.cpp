/**************************************************************
** Example04BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/textShader", [&](){ shape = new XGLTextureAtlas(); return shape; });
}
