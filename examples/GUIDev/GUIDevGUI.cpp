#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiCanvas *g, *g2;

	AddGuiShape("shaders/000-simple", [&]() { return new XGLTransformer(); });

	AddGuiShape("shaders/ortho-tex", [&]() { g = new XGLGuiCanvas(this, 0, 0, 360, 640); return g; });
	g->model = glm::translate(glm::mat4(), glm::vec3(20, 20, 0.0));
	g->RenderText(L"Test");
	g->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.0f, %0.0f\n", s->name.c_str(), x, y);
		return true;
	});

	CreateShape(&guiShapes, "shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 250, 80); return g2; });
	g2->model = glm::translate(glm::mat4(), glm::vec3(10, 70, 0.0));
	g2->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.8 };
	g2->RenderText(L"Another");
	g->AddChild(g2);
	g2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.0f, %0.0f\n", s->name.c_str(), x, y);
		return true;
	});

	XInput::XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && GuiIsActive())
			RenderGui(false);
		else if (isDown)
			RenderGui(true);
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);
}
