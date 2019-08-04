#include "ExampleXGL.h"
#include "xglimgui.h"

class XGLImGui : public XGLTexQuad {
public:
	// For reasons unknown, my icon files are not parsed correctly by SOIL... forceChannels says
	// "I know how many channels there are, make it work."  Hence the "4" as the 2nd arg to XGLTexQuad ctor
	XGLImGui(std::string fn) : XGLTexQuad(fn, 4) {
		SetMadStyleTheme();
	}

	void Draw() {
		ImGui::ShowDemoWindow();

		if (ImGui::Begin("Icons", &demoWindow)) {
			if (ImGui::CollapsingHeader("Controls")) {
				ImGui::SliderInt("Column", &iconX, 0, 13);
				ImGui::SliderInt("Row", &iconY, 0, 14);
				ImGui::ColorEdit4("Tint", &iconTint.x);
				ImGui::SliderInt("Size/32)", &iconPreviewSize, 1, 8);
			}

			if (ImGui::CollapsingHeader("Preview")) {
				ImGui::Image((ImTextureID)texIds[0],
					ImVec2(iconPreviewSize*32.0f, iconPreviewSize*32.0f),
					ImVec2(iconX*dX, iconY*dY),
					ImVec2(iconX*dX + dX, iconY*dY + dY),
					iconTint,
					ImVec4(0.7, 0.3, 0.3, 1.0));
			}

			if (ImGui::CollapsingHeader("Atlas")) {
				ImGui::Image((ImTextureID)texIds[0],
					ImVec2(512, 512),
					ImVec2(0,0),
					ImVec2(1,1),
					iconTint);
			}

			ImGui::End();
		}
		else
			ImGui::End();
	}

	void SetMadStyleTheme() {
		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
		colors[ImGuiCol_TextDisabled] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
		colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
		colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_PopupBg] = ImVec4(0.27f, 0.27f, 0.27f, 0.98f);
		colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
		colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_FrameBg] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
		colors[ImGuiCol_FrameBgHovered] = ImVec4(0.41f, 0.41f, 0.41f, 0.52f);
		colors[ImGuiCol_FrameBgActive] = ImVec4(0.39f, 0.39f, 0.39f, 0.67f);
		colors[ImGuiCol_TitleBg] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
		colors[ImGuiCol_TitleBgActive] = ImVec4(0.37f, 0.21f, 0.21f, 1.00f);
		colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.56f, 0.14f, 0.14f, 0.51f);
		colors[ImGuiCol_MenuBarBg] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
		colors[ImGuiCol_ScrollbarBg] = ImVec4(0.32f, 0.32f, 0.32f, 0.53f);
		colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.58f, 0.58f, 0.58f, 0.80f);
		colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.73f, 0.73f, 0.73f, 0.80f);
		colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.86f, 0.19f, 0.19f, 0.86f);
		colors[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
		colors[ImGuiCol_SliderGrab] = ImVec4(0.21f, 0.21f, 0.21f, 0.78f);
		colors[ImGuiCol_SliderGrabActive] = ImVec4(0.70f, 0.70f, 0.70f, 0.60f);
		colors[ImGuiCol_Button] = ImVec4(0.58f, 0.58f, 0.58f, 0.40f);
		colors[ImGuiCol_ButtonHovered] = ImVec4(0.46f, 0.46f, 0.46f, 1.00f);
		colors[ImGuiCol_ButtonActive] = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
		colors[ImGuiCol_Header] = ImVec4(0.72f, 0.21f, 0.21f, 0.31f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.98f, 0.26f, 0.26f, 0.80f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.98f, 0.26f, 0.26f, 1.00f);
		colors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 0.62f);
		colors[ImGuiCol_SeparatorHovered] = ImVec4(0.80f, 0.14f, 0.14f, 0.78f);
		colors[ImGuiCol_SeparatorActive] = ImVec4(0.80f, 0.14f, 0.14f, 1.00f);
		colors[ImGuiCol_ResizeGrip] = ImVec4(0.30f, 0.30f, 0.30f, 0.56f);
		colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.57f, 0.17f, 0.17f, 0.67f);
		colors[ImGuiCol_ResizeGripActive] = ImVec4(0.78f, 0.20f, 0.20f, 0.95f);
		colors[ImGuiCol_Tab] = ImVec4(0.30f, 0.30f, 0.30f, 0.93f);
		colors[ImGuiCol_TabHovered] = ImVec4(0.98f, 0.26f, 0.26f, 0.80f);
		colors[ImGuiCol_TabActive] = ImVec4(0.40f, 0.22f, 0.22f, 1.00f);
		colors[ImGuiCol_TabUnfocused] = ImVec4(0.92f, 0.93f, 0.94f, 0.99f);
		colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.91f, 0.91f, 0.91f, 1.00f);
		colors[ImGuiCol_PlotLines] = ImVec4(0.71f, 0.67f, 0.02f, 1.00f);
		colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.96f, 1.00f, 0.16f, 1.00f);
		colors[ImGuiCol_PlotHistogram] = ImVec4(0.58f, 0.60f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.95f, 1.00f, 0.00f, 1.00f);
		colors[ImGuiCol_TextSelectedBg] = ImVec4(0.98f, 0.26f, 0.26f, 0.35f);
		colors[ImGuiCol_DragDropTarget] = ImVec4(0.98f, 0.26f, 0.26f, 0.95f);
		colors[ImGuiCol_NavHighlight] = ImVec4(0.98f, 0.26f, 0.26f, 0.80f);
		colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
		colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

	}

private:
	bool demoWindow{ true };

	// the Icon files are arranged in a 14 x 15 grid.
	const float dX{ 1.0f / 14.0f };
	const float dY{ 1.0f / 15.0f };
	int iconX{ 0 };
	int iconY{ 0 };
	ImVec4 iconTint{ 0.5, 0.5, 0.5, 1.0 };
	int iconPreviewSize{ 4 };
};


void ExampleXGL::BuildGUI() {
	XGLGuiManager* gm;
	XGLImGui* im;

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

	std::string imgPath = pathToAssets + "/assets/icons-64.png";

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/zzz", [&](){ im = new XGLImGui(imgPath); return im; });
	im->attributes.diffuseColor = XGLColors::red;

	ImGuiStyle& igStyle = ImGui::GetStyle();
	ImVec4* colors = igStyle.Colors;

	igStyle.WindowRounding = 4.0f;
	igStyle.ChildRounding = 4.0f;
	igStyle.PopupRounding = 4.0f;
	igStyle.FrameRounding = 4.0f;
	igStyle.ScrollbarRounding = 4.0f;
	igStyle.GrabRounding = 3.0f;
	igStyle.TabRounding = 4.0f;

	igStyle.WindowBorderSize = 1.0f;
	igStyle.ChildBorderSize = 1.0f;
	igStyle.PopupBorderSize = 1.0f;
	igStyle.FrameBorderSize = 1.0f;
	igStyle.TabBorderSize = 1.0f;

	return;
}
