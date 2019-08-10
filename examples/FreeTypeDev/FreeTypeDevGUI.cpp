#include "ExampleXGL.h"
#include "xglimgui.h"

bool showGui{ true };

void ExampleXGL::BuildGUI() {
	XGLGuiManager* gm;
	XGLImGui* xig;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });
	gm->AddChildShape("shaders/zzz", [&](){ xig = new XGLImGui(); return xig; });

	xig->AddMenuFunc([&]() {
		if (ImGui::Begin("Titler")) {
			if (ImGui::CollapsingHeader("Controls")) {
			}
			ImGui::End();
		}
		else
			ImGui::End();

		return;
	});
	return;
}
