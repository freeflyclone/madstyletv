/**************************************************************
** Example06BuildScene.cpp
**
** Experiments with XGLSled...
**
** XGLSled is an XGLShape with some aircraft-style attitude
** control methods.
**************************************************************/
#include "ExampleXGL.h"

XGLSled *sled;
XGLShape *cube;

void ExampleXGL::BuildScene() {
	// Add new sled to the scene.  "true" to the constructor turns on display of orientation axes.
	CreateShape("shaders/000-simple", [&](){ sled = new XGLSled(true); return sled; });
	rootShape->AddChild(sled);

	// use the left stick to control yaw, right stick to control pitch & roll of the sled
	// XGLSled::SampleInput(float yaw, float pitch, float roll) also calls XGLSled::GetFinalMatrix()
	AddProportionalFunc("Xbox360Controller0", [this](float v) { rootShape->SampleInput(-v, 0.0f, 0.0f); });
	AddProportionalFunc("Xbox360Controller2", [this](float v) { rootShape->SampleInput(0.0f, 0.0f, v); });
	AddProportionalFunc("Xbox360Controller3", [this](float v) { rootShape->SampleInput(0.0f, -v, 0.0f); });

	CreateShape("shaders/diffuse", [&]() { cube = new XGLCube(); cube->model = glm::scale(glm::mat4(), glm::vec3(0.2, 1.0, 0.01)); return cube; });
	sled->AddChild(cube);
}
