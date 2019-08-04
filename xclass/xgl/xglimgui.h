#ifndef XGLIMGUI_H
#define XGLIMGUI_H

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class XGLImGui : public XGLTexQuad {
public:
	XGLImGui();
	void Draw();
	void SetMadStyleTheme();

private:
	bool demoWindow{ true };

	// the Icon files are arranged in a 14 x 15 grid.
	const float dX{ 1.0f / 14.0f };
	const float dY{ 1.0f / 15.0f };
	int iconX{ 0 };
	int iconY{ 0 };
	ImVec4 iconTint{ 0.75, 0.75, 0.75, 1.0 };
	int iconPreviewSize{ 4 };
};


#endif