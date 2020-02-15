#include "ExampleXGL.h"
#include "xglimgui.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager* gm;

	menuFunctions.push_back([this]() {
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("Configuration"))
			{
				if (ImGui::MenuItem("Load", "CTRL+Z")) {
					xprintf("Load was selected\n");
				}
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
	});

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });
	return;
}
