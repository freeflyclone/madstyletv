/**************************************************************
** SDFTestBuildScene.cpp
**
** Demonstrate Signed Distance Function text rendering.
** Default camera manipulation via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

class SDFfont {
public:
	SDFfont();
	virtual ~SDFfont();
};

SDFfont::SDFfont() {
	xprintf("SDFfont::SDFfont()\n");
}

SDFfont::~SDFfont() {
	xprintf("SDFfont::~SDFfont()\n");
}

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	std::string imgPath = pathToAssets + "/assets/Arial Regular.sdff.png";

	AddShape("shaders/sdf", [&](){ shape = new XGLTexQuad(imgPath); return shape; });
	shape->SetColor({ 1,1,1,0 });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0, 0.0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;
}
