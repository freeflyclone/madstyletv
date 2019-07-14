/**************************************************************
** ImGuiDevBuildScene.cpp
**
** ImGui is a 3rd-party GUI library with tremendous appeal for
** me:  I REALLY don't want to write a GUI layer, because
** writing GUI widgets is way too tedious. ImGui looks like
** it can be made to be pretty enough for professional looking
** UI experiences, which I care about.
**************************************************************/
#include "ExampleXGL.h"

bool guiIsActive{ false };

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
