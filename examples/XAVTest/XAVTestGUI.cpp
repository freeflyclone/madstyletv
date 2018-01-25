#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiWindow *gw;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "GuiTextWindow", 20, 20, 440, 600); return gw; });
	gw->attributes.diffuseColor = XGLColors::yellow;
	gw->SetPenPosition(10, 20);
	gw->RenderText("This window is pinned to the upper left corner. (the default)\nThis is a test, just to see if this works.\n", 16);

	return;
}
