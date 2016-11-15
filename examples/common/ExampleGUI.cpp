#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLShape *shape;
	XGLGuiCanvas *child1,*child2;
	XGLShape *child3;
	glm::mat4 translate, model;

	// add a GUI layer. This is essentially the same as the world layer.
	// With just a "model" transform in the vertex shader, it becomes 2D
	AddGuiShape("shaders/gui", [&]() { shape = new XGLTransformer(); return shape; });

	XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && GuiIsActive())
			RenderGui(false);
		else if (isDown)
			RenderGui(true);
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(1280, 720); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	child1->RenderText(L"Now is the time for all good men \nto come to the aid of their country.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child2 = new XGLGuiCanvas(1280, 720); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, -0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	child1->AddChild(child2);
	child2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return false;
	});

	child2->RenderText(L"Smaller canvas window inside larger one.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(640, 360); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->RenderText(L"Really BIG text.\nReally really big text.\nSeriously big.\nSERIOUSLY! It's big.\nHuge even.");
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(1920, 1080); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0., -0.8f, 0));
	model = glm::scale(translate, glm::vec3(0.99, 0.05, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});
}
