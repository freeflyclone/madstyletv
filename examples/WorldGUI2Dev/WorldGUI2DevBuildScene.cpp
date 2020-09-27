/**************************************************************
** WorldGUI2DevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground" plane and an
** XGLGuiCanvas for VR GUI development.  Heavily commented as
** a product of reverse-engineering my own work ;)
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLImGui *xig;

	AddShape("shaders/zzz", [&]() { xig = new XGLImGui(); return xig; });

	AddShape("shaders/diffuse", [&](){ shape = new XGLSphere(1.0, 32); return shape; });

	shape->model = glm::translate(glm::mat4(), glm::vec3(0.0, -5.0, 0.0));
}
