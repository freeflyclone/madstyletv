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

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	XGLObjectPtr op;

	if ((op = GetGuiManager()->FindObject("HorizontalSlider0")) != nullptr) {
		if (dynamic_cast<XGLGuiCanvas *>((XGLShape *)op)) {
			XGLGuiCanvas *gc = (XGLGuiCanvas *)op;
			gc->AddMouseEventListener([gc](float x, float y, int flags) {
				XGLGuiCanvas *thumb = (XGLGuiCanvas *)gc->Children()[0];
				float xScaled = thumb->model[3][0] / (gc->width - thumb->width) * 100.0f;
				static float previousXscaled = 0.0;

				if (xScaled != previousXscaled) {
					xprintf("%s, %0.4f\n", gc->Name().c_str(), xScaled);
					previousXscaled = xScaled;
				}
			});
		}
	}
	GetGuiManager()->DumpChildren();
}
