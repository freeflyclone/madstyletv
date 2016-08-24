/**************************************************************
** Example02: demonstrate instantiation of a "ground"
** plane and multiple toruses with a diffuse shader
** and transformations.
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/lighting", [&](){ shape = new XGLTorus(4.0f, 1.0f, 64, 32); return shape; });
	shape->SetColor(yellow);


	AddShape("shaders/diffuse", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->SetColor(blue);
	shape->model = glm::translate(glm::mat4(), glm::vec3(10, 0, 0));

	AddShape("shaders/diffuse", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->SetColor(red);
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0)) 
				 * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	AddShape("shaders/diffuse", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->SetColor(green);
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0)) 
				 * glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0,1,0)) 
				 * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));


	XInputKeyFunc renderMod = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool wireFrameMode = FALSE;

		if (isDown && !isRepeat){
			wireFrameMode = wireFrameMode ? false : true;
			glPolygonMode(GL_FRONT_AND_BACK, wireFrameMode ? GL_LINE : GL_FILL);
			GL_CHECK("glPolygonMode() failed.");
		}
	};

	AddKeyFunc('M', renderMod);
	AddKeyFunc('m', renderMod);
}