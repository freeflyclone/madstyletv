/**************************************************************
** WorldGUI2DevBuildScene.cpp
**************************************************************/
#include "ExampleXGL.h"
#include "imgui_stdlib.h"
#include "xsocket.h"


#include <queue>

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


#define FUNCIN(...) xprintf("%s(%d) -->\n", __FUNCTION__, std::this_thread::get_id())
#define FUNCOUT(...) xprintf("%s(%d) <--\n", __FUNCTION__, std::this_thread::get_id())
#define FUNC(...) { xprintf("%s(%d): ", __FUNCTION__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }
#define FLESS(...) { xprintf(" %s:%d (%d) : ", __FILE__, __LINE__, std::this_thread::get_id()) ; xprintf(__VA_ARGS__); }

class XSemaphore2 {
public:
	// Initially, this semaphore must be notify()'d before a wait() will be satisfied
	XSemaphore2(int c = 0) : count(c) {
		FUNCIN();
		FUNCOUT();
	};

	void operator()(int c) {
		FUNCIN();
		std::unique_lock<std::mutex> lock(mutex);
		count = c;
		FUNCOUT();
	}

	void notify(int n = 1) {
		FUNCIN();
		std::unique_lock<std::mutex> lock(mutex);
		count += n;
		cv.notify_one();
		FUNCOUT();
	}

	void wait(int n = 1) {
		FUNCIN();
		std::unique_lock<std::mutex> lock(mutex);
		waitNum = n;
		cv.wait(lock, [this] { return (count >= waitNum); });

		count -= waitNum;
		FUNCOUT();
	}

	bool wait_for(int dlyInMillis, int n = 1) {
		FUNCIN();
		std::unique_lock<std::mutex> lock(mutex);
		waitForNum = n;
		bool retVal = cv.wait_for(lock, std::chrono::milliseconds(dlyInMillis), [this] { return (count >= waitForNum); });

		// if retVal is true, it means we didn't time out.
		if (retVal)
			count -= waitForNum;

		FUNCOUT();
		return retVal;
	}

	unsigned int get_count() {
		FUNCIN();
		std::unique_lock<std::mutex> lock(mutex);
		FUNCOUT();
		return count;
	}

private:
	unsigned int count;
	unsigned int waitNum;
	unsigned int waitForNum;
	std::mutex mutex;
	std::condition_variable cv;
};

class XDispatchQueue : public XThread {
	typedef std::function<void()> Function;
	typedef std::queue<Function> Queue;

public:
	XDispatchQueue() : XThread("DispatchQueue") {
		FUNCIN();
		FUNCOUT();
	};

	~XDispatchQueue() {
		FUNCIN();
		Stop();
		FUNCOUT();
	}

	void operator += (Function fn) {
		FUNCIN();
		std::unique_lock<std::mutex> lock(m_lock);
		m_q.push(fn);
		lock.unlock();
		m_signal.notify();
		FUNCOUT();
	};
	
	void Run() {
		FUNCIN();
		while (IsRunning()) {
			m_signal.wait();
			if (m_q.size())	{
				std::unique_lock<std::mutex> lock(m_lock);
				Function fn = m_q.front();

				m_q.pop();
				fn();
			}
		}
		FUNCOUT();
	}

private:
	std::mutex m_lock;
	XDispatchQueue::Queue m_q;
	XSemaphore m_signal;
};

XDispatchQueue xgdc;

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
					xgdc += [&]() {	FLESS("\"Fire APC\" Button Pushed\n"); };
					//xsh.m_signal.notify();
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

	xgdc.Start();
	FUNC("tid is: %d\n", tid);
}
