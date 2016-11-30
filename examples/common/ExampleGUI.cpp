#include "ExampleXGL.h"

class GuiMain : public XGLGuiCanvas {
public:
	GuiMain(XGL *xgl, bool addTexture = false) : XGLGuiCanvas(xgl, 1, 1, addTexture), padding(20) {
		SetName("GuiMain");

		attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.0 };

		model = glm::translate(glm::mat4(), glm::vec3(padding, padding, 0.0));

		SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
			xprintf("In MouseFunc() for %s : %0.0f, %0.0f\n", s->name.c_str(), x, y);
			return true;
		});

		xgl->projector.AddReshapeCallback(std::bind(&GuiMain::Reshape, this, _1, _2));
	}

	void Reshape(int w, int h) {
		// here we need to adjust the width & height of this shape to the dimensions of the window to trap the mouse events.
		// ...AND adjust the dimensions of the rectangle in the vertices themselves.
		width = w - 2*padding;
		height = h - 2*padding;
		XGLTexQuad::Reshape(0, 0, w - 2*padding, h - 2*padding);
	}

	int padding;
	XGLGuiCanvas *gc;
};

void ExampleXGL::BuildGUI() {
	XGLGuiCanvas *g, *g2, *g3;
	glm::mat4 translate, scale, model;

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

	// this is here just to create a single shape as the root of the XGLGuiCanvas tree,
	// for the ease of GuiResolve() method of XGL;
	AddGuiShape("shaders/000-simple", [&]() { return new XGLTransformer(); });

	AddGuiShape("shaders/ortho", [&]() { g = new GuiMain(this); return g; });

	CreateShape(&guiShapes, "shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 360, 640); return g2; });
	g2->model = glm::translate(glm::mat4(), glm::vec3(800, 20, 0.0));
	g2->RenderText(L"Test");
	g2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.0f, %0.0f\n", s->name.c_str(), x, y);
		return true;
	});
	g->AddChild(g2);

	CreateShape(&guiShapes, "shaders/ortho-tex", [&]() { g3 = new XGLGuiCanvas(this, 250, 80); return g3; });
	g3->model = glm::translate(glm::mat4(), glm::vec3(10, 70, 0.0));
	g3->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.8 };
	g3->RenderText(L"Another");
	g3->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.0f, %0.0f\n", s->name.c_str(), x, y);
		return true;
	});
	g2->AddChild(g3);

	// add the AntTweakBar shape on top of the XGLGuiCanvasWithReshape
	AddGuiShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });

	return;
}
