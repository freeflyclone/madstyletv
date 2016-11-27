/**************************************************************
** ComputeShaderTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

XGLShader *computeShader;

void ExampleXGL::BuildScene() {
	XGLGuiCanvas *shape;
	glm::mat4 translate, scale, rotate;

	computeShader = new XGLShader("shaders/compute-shader");
	computeShader->CompileCompute(pathToAssets + "/shaders/compute-shader");

	AddShape("shaders/tex", [&](){ shape = new XGLGuiCanvas(512,512); return shape; });
	scale = glm::scale(glm::mat4(), glm::vec3(5.0f,5.0f,1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 5.0f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
	shape->model = translate*rotate*scale;
	shape->SetColor({ 1, 1, 1, 0.5 });
	shape->Fill(255);
}
