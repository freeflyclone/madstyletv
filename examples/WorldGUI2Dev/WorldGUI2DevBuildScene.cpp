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

/**
* Simple dispatch queue for async Function object execution on alternate thread(s)
* FIFO implemented with std::queue. 
*
* operator += overload adds a Function to the Queue
* operator & overload pops a Function from Queue, or an empty Function if Queue is empty
*
* Run() method waits on an XSemaphore() then pops a Function and runs it in a separate thread.
*
* One must call XDispatchQueue::Start() to start the thread for background dequeueing & call of Function.
*
* Use of the & overload can be leveraged by let's say an XGLShape::SetAnimationFunc() for 
* dequeueing & call of Function in XGL::Display() thread. (main).
*
* Use of both simultaneously is undefined.
*/
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
		m_queue.push(fn);
		lock.unlock();
		m_signal.notify();
	};

	Function& operator& () {
		if (m_queue.size()) {
			std::unique_lock<std::mutex> lock(m_lock);
			m_fn = m_queue.front();
			m_queue.pop();
			lock.unlock();
			return m_fn;
		}
		else
			return m_emptyFn;
	}

	void Run() {
		while (IsRunning()) {
			m_signal.wait();
			if (m_queue.size()) {
				std::unique_lock<std::mutex> lock(m_lock);
				Function fn = m_queue.front();

				m_queue.pop();
				fn();
			}
		}
	}

private:
	std::mutex m_lock;
	XDispatchQueue::Queue m_queue;
	XSemaphore m_signal;
	Function m_emptyFn{ []() {}  };
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
	char replyRaw[2048];
	char replyCooked[4096];
	void CookHttpRequest() {
		char *s, *d;
		for (s = requestRaw, d = requestCooked; *s && (s-requestRaw < sizeof(requestRaw)-1); s++, d++) {
			if (*s == '\n')
				*d++ = '\r';
			*d = *s;
		}
		*d = 0;
	}

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

		// Menu function runs once per display refresh.
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

					// line-ending converted (LF -> CRLF)
					CookHttpRequest();
					std::string request(requestCooked);

					// we're running in display loop (main) thread
					int nWritten = xsock.Send(request.c_str(), request.size());
					if (console)
					{
						// 
						if (nWritten == request.size())
						{
							console->RenderText("Sent " + std::to_string(nWritten) + " bytes\n");

							// gxdq run it's own background thread 
							gxdq += [&]() {
								FLESS("Send wrote %d bytes\n", nWritten);
								int nRead = xsock.Recv(replyRaw, sizeof(replyRaw));
								FLESS("Got %d bytes back", nRead);

								// lambda: xig's Animation() func runs this, ie: main thread.
								xig->xdq += [&]() {
									console->RenderText(replyRaw);
								};
							};
						}
						else
							console->RenderText("xsock.Send() failed, sent " + std::to_string(nWritten) + " bytes\n");
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

	gxdq.Start();
	xprintf("Main TID: %d\n", tid);
}
