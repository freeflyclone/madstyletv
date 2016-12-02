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

	if ((op = GetGuiRoot()->FindObject("HorizontalSlider0")) != nullptr) {
		if (dynamic_cast<XGLGuiCanvas *>((XGLShape *)op))
			xprintf("Found the slider in %s, and its an XGLGuiCanvas!\n", GetGuiRoot()->name.c_str());
	}
}
