#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiCanvas *child1, *child2, *child3, *g, *g2;
	glm::mat4 translate, scale, model;

	// this is here just to create a single shape as the root of the XGLGuiCanvas tree,
	// for the ease of GuiResolve() method of XGL;
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

	// add the AntTweakBar shape on top of the XGLGuiCanvasWithReshape
	AddGuiShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });

	return;
}
