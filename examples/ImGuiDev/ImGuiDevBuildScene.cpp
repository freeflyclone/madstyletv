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
		pImGuiIO = &io;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows

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

		// Update and Render additional Platform Windows
		// (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
		//  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
		if (pImGuiIO->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}
	}

private:
	XGL* pXgl{ nullptr };
	bool show_demo_window = true;
	ImGuiIO* pImGuiIO{ nullptr };
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLImGui *pImGui;
	XGLGuiManager* gm = GetGuiManager();;


	gm->AddChildShape("shaders/zzz", [&](){ pImGui = new XGLImGui(this); return pImGui; });
}
