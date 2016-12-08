/**************************************************************
** ComputeShaderTestBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader and rendering its output as
** a texture map.
**************************************************************/
#include "ExampleXGL.h"

namespace {
	float roll = 0.1f;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	glm::mat4 translate, scale, rotate;

	// create an XGLTexQuad() without a texture for output of computeShader.
	// We'll add the texture with direct OpenGL calls here, because we'll 
	// be using a format that AddTexture() doesn't know how to create.
	AddShape("shaders/csdraw", [&](){ shape = new XGLTexQuad(); return shape; });
	scale = glm::scale(glm::mat4(), glm::vec3(5.0f,5.0f,1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 5.0f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
	shape->model = translate*rotate*scale;
	shape->attributes.diffuseColor = { 1, 1, 1, 1.0 };
	
	// make the custom format texture and add it to the XGLTexQuad
	GLuint texHandle;
	glGenTextures(1, &texHandle);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	shape->AddTexture(texHandle);

	// Because we'll also using this tex as an image (in order to write to it
	// in the compute-shader), we bind it to an image unit as well
	glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
	GL_CHECK("Gen texture");

	// create the compute shader program object
	XGLShader *computeShader = new XGLShader("shaders/compute-shader");
	computeShader->CompileCompute(pathToAssets + "/shaders/compute-shader");

	// and cause it to be "dispatched" in the preRender phase
	shape->preRenderFunction = [computeShader](float clock) {
		glUseProgram(computeShader->programId);
		glUniform1f(glGetUniformLocation(computeShader->programId, "roll"), (float)clock*roll);
		glDispatchCompute(512 / 16, 512 / 16, 1); // 512^2 threads in blocks of 16^2
		GL_CHECK("Dispatch compute shader");
	};

	// here is where the GUI gets hooked up to actual code.
	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(GetGuiManager()->FindObject("SliderWindow"));
	if (sliders != nullptr) {
		XGLGuiCanvas *vs = (XGLGuiCanvas *)sliders->FindObject("Roll Rate");
		if (vs != nullptr) {
			vs->AddMouseEventListener([vs, computeShader](float x, float y, int flags) {
				XGLGuiCanvas *thumb = (XGLGuiCanvas *)vs->Children()[1];
				float yScaled = ((vs->height - thumb->height) - (thumb->model[3][1])) / (vs->height - thumb->height);
				static float previousYscaled = 0.0;

				if (yScaled != previousYscaled && vs->HasMouse()) {
					roll = yScaled;
					previousYscaled = yScaled;
				}
			});
		}
	}
}
