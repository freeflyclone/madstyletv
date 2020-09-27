/**************************************************************
** WorldGUI2DevBuildScene.cpp
**************************************************************/
#include "ExampleXGL.h"
#include "imgui_stdlib.h"
#include "xsocket.h"

namespace {
	class ImGuiMenu : public XGLImGui {
	public:
		bool show = true;
		std::string hostName = "madstyle.social";
		char ipAddr[64];
	};

	ImGuiMenu *xig = nullptr;
	XGLGuiCanvas *canvas = nullptr;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	glm::vec3 cameraPosition(0, -20, 5.4f);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition * -1.0f + glm::vec3(0, 0, 6.4));
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/diffuse", [&]() {
		shape = new XGLSphere(1.0, 32); return shape;
		shape->model = glm::translate(glm::mat4(), glm::vec3(0.0, -5.0, 0.0));
	});

	AddShape("shaders/diffuse", [&]() { 
		xig = new ImGuiMenu();

		xig->AddMenuFunc(([&]() {
			if (ImGui::Begin("Socket Noodler", &xig->show))
			{
				ImGui::Text("Host name: ");
				ImGui::SameLine();
				ImGui::InputText("", &xig->hostName);
				ImGui::SameLine();
				if (ImGui::Button("Lookup")) {
					xprintf("Lookup pressed: host name is: %s\n", xig->hostName.c_str());
					std::string addr = XSocket::Host2Addr(xig->hostName);
					if (canvas)
					{
						canvas->Clear();
						canvas->RenderText("Address of \"" + xig->hostName + "\" is: " + addr + "\n");
					}
				}
			}
			ImGui::End();
		}));
		return xig;
	});

	if( (canvas = (XGLGuiCanvas*)FindObject("SocketStuff")) != nullptr)
		canvas->RenderText("Socket test...\n");
}
