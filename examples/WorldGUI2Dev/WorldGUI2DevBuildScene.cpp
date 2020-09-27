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
		std::string hostName = "hq.e-man.tv";
		std::string ipAddr;
		int port = 80;
		int type = SOCK_STREAM;
		int proto = IPPROTO_TCP;
		int bindFlag = false;
	};

	ImGuiMenu *xig = nullptr;
	XGLGuiCanvas *canvas = nullptr;
	XGLGuiCanvas *console = nullptr;
	XSocket xsock;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	glm::vec3 cameraPosition(0, -25, 5.4f);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition * -1.0f + glm::vec3(5, 0, 6.4));
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
					xig->ipAddr = XSocket::Host2Addr(xig->hostName);
					if (canvas)
					{
						canvas->Clear();
						canvas->RenderText("Address of \"" + xig->hostName + "\" is: " + xig->ipAddr + "\n");
					}
				}

				ImGui::Text("Open params: ");
				ImGui::SameLine();
				ImGui::Text("addr: %s", xig->ipAddr.c_str());
				ImGui::SameLine();
				ImGui::Text("port: %d", xig->port);
				ImGui::SameLine();
				ImGui::Text("type: %d", xig->type);
				ImGui::SameLine();
				ImGui::Text("proto: %d", xig->proto);
				ImGui::SameLine();
				ImGui::Text("flag: %d", xig->bindFlag);
				ImGui::SameLine();
				if (ImGui::Button("Open")) {
					xprintf("Open clicked\n");
					int retVal = xsock.Open(xig->ipAddr, xig->port);
					if (console)
					{
						console->Clear();
						if (retVal != -1)
							console->RenderText("Socket opened!\n");
						else
							console->RenderText("Socket did not open\n");
					}

				}
				ImGui::SameLine();
				if (ImGui::Button("Close")) {
					xprintf("Close clicked\n");
					int ret = xsock.Close();
					if (console) {
						if (ret == 0)
							console->RenderText("Socket closed.\n");
						else
							console->RenderText("xsock.Close() failed\n");
					}
				}

				ImGui::Text("Connect params: ");
				ImGui::SameLine();
				ImGui::Text("addr: %s", xig->ipAddr.c_str());
				ImGui::SameLine();
				ImGui::Text("port: %d", xig->port);
				ImGui::SameLine();
				if (ImGui::Button("Connect")) {
					xprintf("Connect clicked\n");
					int ret = xsock.Connect();
					if (console)
					{
						if (ret == 0)
							console->RenderText("Socket connected.\n");
						else
							console->RenderText("xsock.Connect() failed\n");
					}
				}
			}
			ImGui::End();
			xsock.GetLastError();
		}));
		return xig;
	});

	if( (canvas = (XGLGuiCanvas*)FindObject("SocketStuff")) != nullptr)
		canvas->RenderText("Socket test...\n");
	if ((console = (XGLGuiCanvas*)FindObject("ConsoleOut")) != nullptr)
		xprintf("Found 'ConsoleOut'\n");
}
