/**************************************************************
** WorldGUI2DevBuildScene.cpp
**************************************************************/
#include "ExampleXGL.h"
#include "imgui_stdlib.h"
#include "xsocket.h"
#include "xdispatchq.h"

#define FUNCIN(...) xprintf("%s(%d) -->\n", __FUNCTION__, std::this_thread::get_id())
#define FUNCOUT(...) xprintf("%s(%d) <--\n", __FUNCTION__, std::this_thread::get_id())
#define FUNC(...) { xprintf("%s(%d): ", __FUNCTION__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }
#define FLESS(...) { xprintf(" %s:%d (%d) : ", __FILE__, __LINE__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }

class XHttpClient {
public:
	XHttpClient() {
		if (hostName.size())
			GetHostAddr();
	}

	std::string GetHostAddr() {
		ipAddr = XSocket::Host2Addr(hostName);
		return ipAddr;
	}

	std::string GetHostName() {	return hostName; }
	std::string GetAddr() { return ipAddr; }
	int GetPort() { return port; }
	int GetSocketType() { return type; }
	int GetSocketProto() { return proto; }

	int Open() { return xsock.Open(ipAddr, port); }
	int Close() { return xsock.Close(); }
	int Connect() { return xsock.Connect(); }
	int Send(const char *src, int length) { return xsock.Send(src, length); }
	int Recv(char *dst, int size) { return xsock.Recv(dst, size); }

	int GetLastError() { return xsock.GetLastError(); }

	char *RequestRaw() { return requestRaw; }
	int RequestRawSize() { return sizeof(requestRaw); }
	char *RequestCooked() { return requestCooked; }
	int RequestCookedSize() { return sizeof(requestCooked); }

	char *ReplyRaw() { return replyRaw; }
	int ReplyRawSize() { return sizeof(replyRaw); }
	char *ReplyCooked() { return replyCooked; }
	int ReplyCookedSize() { return sizeof(replyCooked); }

	void CookHttpRequest() {
		char *s, *d;
		for (s = requestRaw, d = requestCooked; *s && (s - requestRaw < sizeof(requestRaw) - 1); s++, d++) {
			if (*s == '\n')
				*d++ = '\r';
			*d = *s;
		}
		*d = 0;
	}

private:
	std::string hostName = "hq.e-man.tv";
	std::string ipAddr;
	int port = 80;
	int type = SOCK_STREAM;
	int proto = IPPROTO_TCP;
	int bindFlag = false;
	XDispatchQueue xdq;
	XSocket xsock;

	char requestRaw[2048]{
		"GET / HTTP/1.1\n"
		"Host: hq.e-man-tv\n"
		"User-Agent: curl/7.54.0\n"
		"Accept: */*\n"
		//"Range: bytes=0-100\n"
		"\n"
	};
	char requestCooked[4096];
	char replyRaw[2048];
	char replyCooked[4096];
};

namespace {
	class ImGuiMenu : public XGLImGui {
	public:
		ImGuiMenu() : XGLImGui() {};
		bool show = true;
		XHttpClient client;
		XDispatchQueue xdq;
	};

	ImGuiMenu *xig = nullptr;
	XGLGuiCanvas *canvas = nullptr;
	XGLGuiCanvas *console = nullptr;

	class MainThreadId {
	public:
		MainThreadId() { tid = std::this_thread::get_id(); }
		std::thread::id operator & () {
			return tid;
		}

	private:
		std::thread::id tid;
	};
	const MainThreadId tid;

};

XDispatchQueue gxdq;
XHttpClient client;
char replyBuff[2048];

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

		// Menu function runs once per display refresh.
		xig->AddMenuFunc(([&]() {
			if (ImGui::Begin("Socket Noodler", &xig->show))
			{
				ImGui::Text("Host name: ");
				ImGui::SameLine();
				ImGui::InputText("", &client.GetHostName());
				ImGui::SameLine();

				if (ImGui::Button("Lookup")) {
					xprintf("Lookup pressed: host name is: %s\n", client.GetHostName().c_str());
					if (canvas)
					{
						canvas->Clear();
						canvas->RenderText("Address of \"" + client.GetHostName() + "\" is: " + client.GetHostAddr() + "\n");
					}
				}

				ImGui::Text("Open params: ");
				ImGui::SameLine();
				ImGui::Text("addr: %s", client.GetAddr().c_str());
				ImGui::SameLine();
				ImGui::Text("port: %d", client.GetPort());
				ImGui::SameLine();
				ImGui::Text("type: %d", client.GetSocketType());
				ImGui::SameLine();
				ImGui::Text("proto: %d", client.GetSocketProto());
				ImGui::SameLine();

				if (ImGui::Button("Open")) {
					xprintf("Open clicked\n");
					int retVal = client.Open();
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
					int ret = client.Close();
					if (console) {
						if (ret == 0)
							console->RenderText("Socket closed.\n");
						else
							console->RenderText("xsock.Close() failed\n");
					}
				}

				ImGui::Text("Connect params: "); 
				ImGui::SameLine();
				ImGui::Text("addr: %s", client.GetAddr().c_str());
				ImGui::SameLine();
				ImGui::Text("port: %d", client.GetPort());
				ImGui::SameLine();
				if (ImGui::Button("Connect")) {
					xprintf("Connect clicked\n");
					int ret = client.Connect();
					if (console)
					{
						if (ret == 0)
							console->RenderText("Socket connected.\n");
						else
							console->RenderText("xsock.Connect() failed\n");
					}
				}

				ImGui::Text("Send params: ");
				ImGui::SameLine();
				ImGui::InputTextMultiline(" ", client.RequestRaw(), client.RequestRawSize());
				ImGui::SameLine();
				if (ImGui::Button("Send")) {
					xprintf("Send clicked\n");

					client.CookHttpRequest();
					std::string request(client.RequestCooked());

					// we're running in display loop (main) thread
					int nWritten = client.Send(request.c_str(), request.size());
					if (console)
					{
						if (nWritten == request.size())
						{
							console->RenderText("Sent " + std::to_string(nWritten) + " bytes\n");

							// gxdq runs it's own background thread
							gxdq.Post( [&]() {
								FLESS("Send wrote %d bytes\n", nWritten);
								int nRead = client.Recv(replyBuff, sizeof(replyBuff));
								if (nRead > 0) {
									FLESS("Got %d bytes back", nRead);

									// lambda: xig's Animation() func runs this, ie: main thread.
									xig->xdq.Post([&]() {
										console->RenderText(replyBuff);
									});
								}
							});
						}
						else
							console->RenderText("xsock.Send() failed, sent " + std::to_string(nWritten) + " bytes\n");
					}
				}
				if (ImGui::Button("Fire APC")) {
					xig->xdq.Post( [&]() {
						console->RenderText("APC fired!\n");
					});
				}
			}
			ImGui::End();
			client.GetLastError();
		}));

		xig->SetAnimationFunction([&](double clock) {
			XDispatchQueue::Function fn = xig->xdq.Remove();
			fn();
		});

		return xig;
	});


	if( (canvas = (XGLGuiCanvas*)FindObject("SocketStuff")) != nullptr)
		canvas->RenderText("Socket test...\n");
	if ((console = (XGLGuiCanvas*)FindObject("ConsoleOut")) != nullptr)
		xprintf("Found 'ConsoleOut'\n");

	gxdq.Start();
	xprintf("Main TID: %d\n", tid);
}
