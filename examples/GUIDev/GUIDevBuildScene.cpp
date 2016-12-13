/**************************************************************
** GUIDevBuildScene.cpp
**
** Demonstrates interaction with the GUI stack. BuildScene()
** is called after BuildGUI() by the ExampleXGL constructor,
** so it is safe to assume it exists at this point.
**
** Here is where we demonstrate adding to the GUI stack, and
** connecting GUI stack objects to world objects for a
** given application.
**************************************************************/
#include "ExampleXGL.h"

class XGLGuiTextEdit : public XGLGuiCanvas {
public:
	XGLGuiTextEdit(XGL *xgl, std::string name, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h){
		SetName(name);
		attributes.ambientColor = XGLColors::black;
		attributes.ambientColor.a = 0.7f;
		attributes.diffuseColor = XGLColors::black;

		AddChildShape("shaders/ortho", [xgl, this, name, x, y]() { label = new XGLGuiLabel(xgl, name, x, y); return label; });
		label->model = glm::translate(glm::mat4(), glm::vec3(-(label->labelWidth + label->labelPadding), 0, 0.0));

		model = glm::translate(glm::mat4(), glm::vec3(x + (label->labelWidth + label->labelPadding), y, 0));
	}

private:
	XGLGuiLabel *label;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLGuiManager *gm = GetGuiManager();
	XGLGuiWindow *gw;
	XGLGuiTextEdit *gte;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 480, 20, 360, 200); return gw; });
	gw->attributes.diffuseColor = { 0.7, 0.7, 0.7, 1.0 };
	gw->attributes.ambientColor = { 0.1, 0.1, 0.1, 0.5 };

	gw->SetPenPosition(10, 20);
	gw->RenderText("Container for XGLGuiTextEdit fields.\n", 20);

	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 1", 20, 40, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 2", 20, 80, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 3", 20, 120, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 4", 20, 160, 200, 24); return gte; });
}
