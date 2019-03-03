#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiWindow *gw;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 20, 20, 440, 60); return gw; });
	gw->attributes.diffuseColor = XGLColors::white;
	gw->SetPenPosition(10, 20);
	//gw->RenderText("This window is pinned to the upper left corner. (the default)\nThis is a test, just to see if this works.\n", 16);
	gw->RenderText("The slider below to controls the number of primitives drawn.\nThis affects the \"count\" arg to glDrawArrays().\n", 16);

	gm->AddChildShape("shaders/ortho", [&]() { gw = new XGLGuiWindow(this, "HorizontalSliderWindow", 20, 100, 1240, 60); return gw; });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Draw Count", XGLGuiSlider::Orientation::horizontal, 20, 20, 1100, 16); });

	return;
}
