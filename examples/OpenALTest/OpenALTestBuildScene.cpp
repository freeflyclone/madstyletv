/**************************************************************
** OpenALTestBuildScene.cpp
**
** This is a copy of Example05, with unit testing of XAL.
**
** This *should* emit a tone for 4 seconds.
**
** If no exceptions are thrown by XAL, but you still don't hear
** anything, it's probably that the default device not chosen
** correctly.
**************************************************************/
#include "ExampleXGL.h"
#include "xal.h"

XAL *pXal;
XAL *pRecXal;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	pXal = new XAL(NULL, XAL::defaultSamplerate, XAL::defaultFormat, XAL::maxBuffers);
	pRecXal = new XAL(NULL);

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";

	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	shape->SetAnimationFunction([shape](float clock) {
		float sinFunc = sin(clock / 40.0f) * 10.0f;
		float cosFunc = cos(clock / 40.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate * scale;
	});
	
	// add all the buffers (they're 1024 samples)
	pXal->AddBuffers(XAL::maxBuffers);
	// initialize them with a test tone
	pXal->TestTone(XAL::maxBuffers);
	// queue the buffers
	pXal->QueueBuffers();
	// and start playing
	pXal->Play();
}
