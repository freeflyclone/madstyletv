#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiCanvas *xgc, *xgc2;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });
	AddMouseFunc([this](int x, int y, int flags) {
		if (GuiIsActive())
			mt.Event(x, y, flags);
	});

	gm->AddChildShape("shaders/gui-tex", [&xgc, this](){
		xgc = new XGLGuiCanvas(this, 1920, 1080);
		xgc->SetName("SocketStuff", false);

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.01, 0.01, 0.01));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-9.6, 0.0, 10.8));

		xgc->model = translate * rotate * scale;
		xgc->attributes.ambientColor = { 0.1, 0.1, 0.1, 0.75 };
		xgc->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };

		KeyEvent('~', 0);

		return xgc;
	});

	gm->AddChildShape("shaders/gui-tex", [&xgc2, this]() {
		xgc2 = new XGLGuiCanvas(this, 1920, 1080);
		xgc2->SetName("ConsoleOut", false);

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.01, 0.01, 0.01));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-9.6, 0.0, 10.8));
		glm::mat4 newModel = translate * rotate * scale;

		glm::mat4 newTranslate = glm::translate(glm::mat4(), glm::vec3(20, -7.5, 0));
		glm::mat4 newRotate = glm::rotate(glm::mat4(), glm::radians(-45.0f), glm::vec3(0.f, 0.0f, 1.0f));

		xgc2->model = newTranslate * newRotate * newModel;

		xgc2->attributes.ambientColor = { 0.1, 0.1, 0.3, 0.75 };
		xgc2->attributes.diffuseColor = { 1.0, 1.0, 0.0, 1.0 };
		xgc2->SetDefaultPixelSize(32);
		xgc2->RenderText("Console Out...\n");

		return xgc2;
	});
}
