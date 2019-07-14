#include "ExampleXGL.h"

extern bool guiIsActive;

void ExampleXGL::BuildGUI() {
	XInput::XInputKeyFunc PresentGuiCanvas = [this](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && guiIsActive) {
			xprintf("RenderGui(false)\n");
			guiIsActive = false;
		}
		else if (isDown) {
			xprintf("RenderGui(true)\n");
			guiIsActive = true;
		}
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);

	return;
}
