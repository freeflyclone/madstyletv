/**************************************************************
** OusterDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and an XGLTextureAtlas, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/textShader", [&](){ shape = new XGLTextureAtlas(); return shape; });
}
