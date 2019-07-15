#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager* gm;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	return;
}
