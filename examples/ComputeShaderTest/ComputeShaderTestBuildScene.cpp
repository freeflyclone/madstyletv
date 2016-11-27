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
	XGLShape *shape;
	glm::mat4 translate, scale, rotate;

	computeShader = new XGLShader("shaders/compute-shader");
	computeShader->CompileCompute(pathToAssets + "/shaders/compute-shader");

	AddShape("shaders/csdraw", [&](){ shape = new XGLTexQuad(); return shape; });
	scale = glm::scale(glm::mat4(), glm::vec3(5.0f,5.0f,1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 5.0f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
	shape->model = translate*rotate*scale;
	shape->SetColor({ 1, 1, 1, 0.5 });

	GLuint texHandle;
	glGenTextures(1, &texHandle);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 512, 0, GL_RED, GL_FLOAT, NULL);

	// Because we're also using this tex as an image (in order to write to it),
	// we bind it to an image unit as well
	glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
	GL_CHECK("Gen texture");

	shape->AddTexture(texHandle);

	shape->preRenderFunction = [&](XGLShape *s, float clock) {
		glUseProgram(computeShader->programId);
		glUniform1f(glGetUniformLocation(computeShader->programId, "roll"), (float)clock*0.01f);
		glDispatchCompute(512 / 16, 512 / 16, 1); // 512^2 threads in blocks of 16^2
		GL_CHECK("Dispatch compute shader");
	};
}
