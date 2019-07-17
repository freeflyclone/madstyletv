#include "ExampleXGL.h"
#include "xglimgui.h"

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

	ImGuiMenuFn fn = [this]() {
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("Your Mom"))
			{
				if (ImGui::MenuItem("Undo", "CTRL+Z")) {
					xprintf("she's been undone");
				}
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
	};

	menuFunctions.push_back(fn);

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/zzz", [&](){ im = new XGLImGui(); return im; });

	return;
}
