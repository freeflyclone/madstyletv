/**************************************************************
** CudaTextureBuildScene.cpp
**
** CUDA/OpenGL example, derived from simpleGL CUDA sample.
**************************************************************/
#include "ExampleXGL.h"
#include "XGLCuda.h"

void ExampleXGL::BuildScene() {
	XGLCuda *shape;
	float scaleFactor = 4.0f;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/tex", [&shape, imgPath, this](){ shape = new XGLCuda(this, imgPath); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	shape->SetAnimationFunction([shape](float clock) {
		shape->RunKernel(clock / 50.000f);
	});

	preferredSwapInterval = 1;
}
