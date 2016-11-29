#include "ExampleXGL.h"
/*
void ExampleXGL::BuildGUI() {
	XGLGuiCanvas *g;

	AddGuiShape("shaders/ortho-tex", [&]() { g = new XGLGuiCanvas(this, -20, 20, 360, 640); return g; });
	g->RenderText(L"Test");

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
*/