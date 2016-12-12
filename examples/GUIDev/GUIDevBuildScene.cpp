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
		attributes.ambientColor = {0.00001, 0.00001, 0.00001, 0.8};
		attributes.diffuseColor = XGLColors::black;
		model = glm::translate(glm::mat4(), glm::vec3(x, y, 0));
	}

private:
	XGLGuiCanvas *label;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLGuiManager *gm = GetGuiManager();
	XGLGuiWindow *gw;
	XGLGuiSlider *hs;
	XGLGuiLabel *gl;
	XGLGuiTextEdit *gte;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 480, 20, 360, 200); return gw; });
	gw->attributes.diffuseColor = XGLColors::yellow;
	gw->SetPenPosition(10, 20);
	gw->RenderText("Container for XGLGuiTextEdit fields.\n", 20);

	gw->AddChildShape("shaders/ortho-tex", [&gl, this]() { gl = new XGLGuiLabel(this, "Test XGLGuiLabel", 10, 40); return gl; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "", 140, 40, 200, 24); return gte; });
}
