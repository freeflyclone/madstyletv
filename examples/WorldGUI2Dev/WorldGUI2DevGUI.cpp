#include "ExampleXGL.h"
#include "xsocket.h"

XSocket xsock;

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiCanvas *shape;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });
	AddMouseFunc([this](int x, int y, int flags) {
		if (GuiIsActive())
			mt.Event(x, y, flags);
	});

	gm->AddChildShape("shaders/gui-tex", [&shape, this](){
		shape = new XGLGuiCanvas(this, 1920, 1080);

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.01, 0.01, 0.01));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-9.6, 0.0, 10.8));

		shape->model = translate * rotate * scale;
		shape->attributes.ambientColor = { 0.01, 0.01, 0.01, 0.5 };
		shape->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };

		KeyEvent('~', 0);

		return shape;
	});

	std::string hostName("www.madstyle.tv");
	std::string addr = XSocket::Host2Addr(hostName);

	shape->RenderText("Socket test...\n");
	shape->RenderText("Address of \"" + hostName + "\" is: " + addr + "\n");
}
