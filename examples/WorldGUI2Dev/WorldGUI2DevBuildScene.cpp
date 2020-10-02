/**************************************************************
** WorldGUI2DevBuildScene.cpp
**************************************************************/
#include "ExampleXGL.h"
#include "imgui_stdlib.h"
#include "xsocket.h"

#include <queue>

#define FUNCIN(...) xprintf("%s(%d) -->\n", __FUNCTION__, std::this_thread::get_id())
#define FUNCOUT(...) xprintf("%s(%d) <--\n", __FUNCTION__, std::this_thread::get_id())
#define FUNC(...) { xprintf("%s(%d): ", __FUNCTION__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }
#define FLESS(...) { xprintf(" %s:%d (%d) : ", __FILE__, __LINE__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }

class XDispatchQueue : public XThread {
	typedef std::function<void()> Function;
	typedef std::queue<Function> Queue;

public:
	XDispatchQueue() : XThread("DispatchQueue") {};

	~XDispatchQueue() {
		Stop();
	}

	void operator += (Function fn) {
		std::unique_lock<std::mutex> lock(m_lock);
		m_q.push(fn);
		lock.unlock();
		m_signal.notify();
	};

	Function& operator& () {
		if (m_q.size()) {
			std::unique_lock<std::mutex> lock(m_lock);
			m_fn = m_q.front();
			m_q.pop();
			return m_fn;
		}
		else
			return m_emptyFn;
	}

	void Run() {
		while (IsRunning()) {
			m_signal.wait();
			if (m_q.size()) {
				std::unique_lock<std::mutex> lock(m_lock);
				Function fn = m_q.front();

				m_q.pop();
				fn();
			}
		}
	}

private:
	std::mutex m_lock;
	XDispatchQueue::Queue m_q;
	XSemaphore m_signal;
	Function m_emptyFn{ []() {} };
	Function m_fn;
};

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
		XDispatchQueue xdq;
	};

	ImGuiMenu *xig = nullptr;
	XGLGuiCanvas *canvas = nullptr;
	XGLGuiCanvas *console = nullptr;
	XSocket xsock;

	char requestRaw[2048] {
		"GET /\n"
		"Host: hq.e-man-tv\n"
		"User-Agent: curl/7.54.0\n"
		"Accept: */*\n"
		"\n" 
	};
	char requestCooked[4096];

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


				ImGui::Text("Send params: ");
				ImGui::SameLine();
				ImGui::InputTextMultiline(" ", requestRaw, sizeof(requestRaw));
				ImGui::SameLine();
				if (ImGui::Button("Send")) {
					xprintf("Send clicked\n");
					char *s, *d;
					for (s = requestRaw, d = requestCooked; *s; s++, d++)
					{
						if (*s == '\n')
							*d++ = '\r';
						*d = *s;
					}
					*d = 0;

					std::string request(requestCooked);

					int ret = xsock.Send(request.c_str(), request.size());
					if (console)
					{
						if (ret == request.size())
							console->RenderText("Sent " + std::to_string(ret) + " bytes\n");
						else
							console->RenderText("xsock.Send() failed, sent " + std::to_string(ret) + " bytes\n");
					}
				}
				if (ImGui::Button("Fire APC")) {
					xig->xdq += [&]() {
						console->RenderText("APC fired!\n");
					};
				}
			}
			ImGui::End();
			xsock.GetLastError();
		}));

		xig->SetAnimationFunction([&](double clock) {
			(&xig->xdq)();
		});

		return xig;
	});


	if( (canvas = (XGLGuiCanvas*)FindObject("SocketStuff")) != nullptr)
		canvas->RenderText("Socket test...\n");
	if ((console = (XGLGuiCanvas*)FindObject("ConsoleOut")) != nullptr)
		xprintf("Found 'ConsoleOut'\n");

	FUNC("tid is: %d\n", tid);
}
