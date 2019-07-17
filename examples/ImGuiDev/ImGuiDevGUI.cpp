#include "ExampleXGL.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class XGLImGui : public XGLShape {
public:
	void Draw() {
		ImGui::ShowDemoWindow(&demoWindow);
	}

private:
	bool demoWindow{ true };
};

void ExampleXGL::BuildGUI() {
	XGLGuiManager* gm;
	XGLImGui* im;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/zzz", [&](){ im = new XGLImGui(); return im; });

	return;
}
