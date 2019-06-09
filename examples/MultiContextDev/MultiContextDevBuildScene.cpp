/**************************************************************
** MultiContextDevBuildScene.cpp
**
** To support asynchronous image upload for multimedia apps,
** multiple OpenGL contexts are required.  Up until now
** XGL has only had the main OpenGL context.  XGLContext
** provides integration of OpenGL context management for
** multi-threaded OpenGL goodness.
**
** Start with a background thread / GL context that repeatedly 
** copies alternating static frames to an oversized "circular"
** PBO that is mapped persistently, so it's always available 
** to CPU code.
**
** The upload thread simulates the output of a video decoding 
** thread from FFMpeg.
**
** The threads "upload thread" and "main rendering thread" use
** OpenGL GLsync objects to stay in sync, thus no tearing
** of the texture image is visible in the output.
**
** XTimer is used for precise timing measurements.
**************************************************************/
#include "ExampleXGL.h"

#include "xglcontextimage.h"

uint8_t* pGlobalPboBuffer;

void ExampleXGL::BuildScene() {
	XGLContextImage *shape;

	// get a new XGLContextImage()
	AddShape("shaders/yuv", [&shape,this](){ shape = new XGLContextImage(this, 1920, 1080, 1); return shape; });

	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	shape->Start();
}
