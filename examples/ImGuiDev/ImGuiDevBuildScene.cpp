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
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

bool guiIsActive{ false };
const char* glsl_version = "#version 150";

class XGLImGui : public XGLShape {
public:
	XGLImGui(ExampleXGL* pXgl) : pXgl(pXgl) {
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(pXgl->window, true);
		ImGui_ImplOpenGL3_Init(glsl_version);

		ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\Arial.ttf", 18.0f);
		IM_ASSERT(font != NULL);
	};

	void Draw() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

private:
	XGL* pXgl{ nullptr };
	bool show_demo_window = true;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLImGui *pImGui;

	AddShape("shaders/zzz", [&](){ pImGui = new XGLImGui(this); return pImGui; });

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
