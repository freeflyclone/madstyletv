/**************************************************************
** ComputeShaderTestBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader and rendering its output as
** a texture map.
**************************************************************/
#include "ExampleXGL.h"

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
		glUniform1f(glGetUniformLocation(computeShader->programId, "roll"), (float)clock*0.05f);
		glDispatchCompute(512 / 16, 512 / 16, 1); // 512^2 threads in blocks of 16^2
		GL_CHECK("Dispatch compute shader");
	};

	XGLGuiManager *gm = GetGuiManager();
	XGLGuiCanvas *g2,*g3,*g4;

	gm->AddChildShape("shaders/ortho", [&]() { g2 = new XGLGuiCanvas(this, 304, 500); return g2; });
	g2->model = glm::translate(glm::mat4(), glm::vec3(200, 104, 0));
	g2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };

	g2->AddChildShape("shaders/ortho", [this,&g3]() { g3 = new XGLGuiCanvas(this, 16, 400); return g3; });
	g3->SetName("VerticalSlider");
	g3->attributes.diffuseColor = { 1, 1, 1, 0.1 };
	g3->model = glm::translate(glm::mat4(), glm::vec3(20.0, 40.0, 0.0));
	g3->SetMouseFunc([this, g3](float x, float y, int flags){
		if (flags & 1) {
			XGLGuiCanvas *slider = (XGLGuiCanvas *)(g3->Children()[1]);
			// constrain mouse Y coordinate to dimensions of track
			float yLimited = (y<0) ? 0 : (y>(g3->height - slider->height)) ? (g3->height - slider->height) : y;
			static float previousYlimited = 0.0;

			if (yLimited != previousYlimited) {
				slider->model = glm::translate(glm::mat4(), glm::vec3(0.0, yLimited, 0.0));
				previousYlimited = yLimited;
			}
			mouseCaptured = g3;
			g3->SetHasMouse(true);
		}
		else {
			mouseCaptured = NULL;
			g3->SetHasMouse(false);
		}
		return true;
	});
	g3->AddChildShape("shaders/ortho", [this, &g4]() { g4 = new XGLGuiCanvas(this, 1, 392, false); return g4; });
	g4->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
	g4->model = glm::translate(glm::mat4(), glm::vec3(8.0, 4.0, 0.0));

	g3->AddChildShape("shaders/ortho-rgb", [this, &g4]() { g4 = new XGLGuiCanvas(this, 16, 16, false); return g4; });
	g4->AddTexture(pathToAssets + "/assets/button.png");
	g4->attributes.diffuseColor = { 1.0, 0.0, 1.0, 0.8 };
	g4->Reshape(0, 0, 16, 16);
	g4->model = glm::translate(glm::mat4(), glm::vec3(0.0, 384.0, 0.0));

	g2->AddChildShape("shaders/ortho", [this, &g3]() { g3 = new XGLGuiCanvas(this, 200, 16); return g3; });
	g3->SetName("HorizontalSlider");
	g3->attributes.diffuseColor = { 1, 1, 1, 0.06 };
	g3->model = glm::translate(glm::mat4(), glm::vec3(20.0, 460.0, 0.0));
	g3->SetMouseFunc([this, g3](float x, float y, int flags){
		if (flags & 1) {
			XGLGuiCanvas *slider = (XGLGuiCanvas *)(g3->Children()[1]);
			// constrain mouse X coordinate to dimensions of track
			float xLimited = (x<0) ? 0 : (x>(g3->width - slider->width)) ? (g3->width - slider->width) : x;
			static float previousXlimited = 0.0;

			if (xLimited != previousXlimited) {
				slider->model = glm::translate(glm::mat4(), glm::vec3(xLimited, 0.0, 0.0));
				previousXlimited = xLimited;
			}
			mouseCaptured = g3;
			g3->SetHasMouse(true);
		}
		else {
			mouseCaptured = NULL;
			g3->SetHasMouse(false);
		}
		return true;
	});
	g3->AddChildShape("shaders/ortho", [this, &g4]() { g4 = new XGLGuiCanvas(this, 192, 1, false); return g4; });
	g4->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	g4->model = glm::translate(glm::mat4(), glm::vec3(4.0, 8.0, 0.0));

	g3->AddChildShape("shaders/ortho-rgb", [this, &g4]() { g4 = new XGLGuiCanvas(this, 16, 16, false); return g4; });
	g4->AddTexture(pathToAssets + "/assets/button.png");
	g4->attributes.diffuseColor = { 1.0, 0.0, 1.0, 1.0 };
	g4->Reshape(0,0,16,16);
}
