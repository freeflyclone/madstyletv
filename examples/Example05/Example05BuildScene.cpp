/**************************************************************
** Example05BuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single texture-mapped quad.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	std::string imgPath = pathToAssets + "/assets/8bit.png";
	AddShape("shaders/textShader", [&](){ shape = new XGLTexQuad(imgPath); return shape; });
}
