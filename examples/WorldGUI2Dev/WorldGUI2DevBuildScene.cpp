/**************************************************************
** WorldGUI2DevBuildScene.cpp
**************************************************************/
#include "ExampleXGL.h"
#include "imgui_stdlib.h"
#include "xsocket.h"
#include "xdispatchq.h"
#include "xlog.h"
#include <string>

// For running running stuff in the background.
XDispatchQueue gxdq;

class ImGuiMenu *xig = nullptr;

class XHttpClient {
public:
	XHttpClient() {
		if (m_hostName.size())
			GetHostAddr();
	}

	std::string GetHostAddr() {
		m_ipAddr = XSocket::Host2Addr(m_hostName);
		return m_ipAddr;
	}

	std::string GetHostName() {	return m_hostName; }
	std::string GetAddr() { return m_ipAddr; }
	int GetPort() { return m_port; }
	int GetSocketType() { return m_type; }
	int GetSocketProto() { return m_proto; }

	int Open() { return m_xsock.Open(m_ipAddr, m_port); }
	int Close() { return m_xsock.Close(); }
	int Connect() { return m_xsock.Connect(); }
	int Send(const char *src, int length) { return m_xsock.Send(src, length); }
	int Recv(char *dst, int size) { return m_xsock.Recv(dst, size); }

	int GetLastError() { return m_xsock.GetLastError(); }

	char *RequestRaw() { return requestRaw; }
	int RequestRawSize() { return sizeof(requestRaw); }
	char *RequestCooked() { return requestCooked; }
	int RequestCookedSize() { return sizeof(requestCooked); }

	char *ReplyRaw() { return replyRaw; }
	int ReplyRawSize() { return sizeof(replyRaw); }
	char *ReplyCooked() { return replyCooked; }
	int ReplyCookedSize() { return sizeof(replyCooked); }

	std::string CookHttpRequest() {
		char *s, *d;
		for (s = requestRaw, d = requestCooked; *s && (s - requestRaw < sizeof(requestRaw) - 1); s++, d++) {
			if (*s == '\n')
				*d++ = '\r';
			*d = *s;
		}
		*d = 0;

		return requestCooked;
	}

	std::string CookHttpReply(char *s) {
		char *start = s;
		char *d = replyCooked;

		for (; s && *s && (s - start < sizeof(replyCooked) - 1); ) {
			if (*s == '\r') {
				s++;
				continue;
			}
			*d++ = *s++;
		}
		*d = 0;

		return replyCooked;
	}
private:
	std::string m_hostName = "hq.e-man.tv";
	std::string m_ipAddr;
	int m_port = 80;
	int m_type = SOCK_STREAM;
	int m_proto = IPPROTO_TCP;
	int m_bindFlag = false;
	//XDispatchQueue m_xdq;
	XSocket m_xsock;

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
	char replyCooked[2048];
};

class ImGuiMenu : public XGLImGui {
public:
	ImGuiMenu(XGL& xgl) : XGLImGui() {
		m_xdq = new XDispatchQueue("ImGuiDispatchQueue");
		AddMenuFunc(std::bind(&ImGuiMenu::Handler, this));

		if ((m_sockwin = (XGLGuiCanvas*)xgl.FindObject("SocketStuff")) != nullptr)
			m_sockwin->RenderText("Socket test...\n");
		if ((m_console = (XGLGuiCanvas*)xgl.FindObject("ConsoleOut")) != nullptr)
			m_console->RenderText("Found 'ConsoleOut'\n");

		SetAnimationFunction([&](double clock) {
			XDispatchQueue::Function fn = m_xdq->Remove();
			fn();
		});

		m_tid = std::this_thread::get_id();
	};

	void Handler() {
		using namespace ImGui;

		if (Begin("Socket Noodler", &m_show)) {
			if (Button("Fire APC")) {
				gxdq.Post([&]() {
					FUNC("APC fired, m_tid: %d.\n", m_tid);
					m_xdq->Post([&]() {
						if (m_console) {
							m_console->RenderText("APC fired: ");
						}
					});
				});
			}
			Text("Host name: ");
			SameLine();
			InputText("", &m_client.GetHostName());
			SameLine();

			if (Button("Lookup")) {
				xprintf("Lookup: host name is: %s\n", m_client.GetHostName().c_str());
				if (m_sockwin)
				{
					m_sockwin->Clear();
					m_sockwin->RenderText("Address of \"" + m_client.GetHostName() + "\" is: " + m_client.GetHostAddr() + "\n");
				}
			}

			Text("Open params: ");    				    SameLine();
			Text("addr: %s", m_client.GetAddr().c_str()); SameLine();
			Text("port: %d", m_client.GetPort());         SameLine();
			Text("type: %d", m_client.GetSocketType());   SameLine();
			Text("proto: %d", m_client.GetSocketProto()); SameLine();

			if (Button("Open")) {
				xprintf("Open clicked\n");
				int retVal = m_client.Open();
				if (m_sockwin) {
					m_sockwin->Clear();
					if (retVal != -1)
						m_sockwin->RenderText("Socket opened!\n");
					else
						m_sockwin->RenderText("Socket did not open\n");
				}
			}
			SameLine();

			if (Button("Close")) {
				xprintf("Close clicked\n");
				int ret = m_client.Close();
				if (m_sockwin) {
					if (ret == 0)
						m_sockwin->RenderText("Socket closed.\n");
					else
						m_sockwin->RenderText("m_xsock.Close() failed\n");
				}
			}

			Text("Connect params: ");
			SameLine();
			Text("addr: %s", m_client.GetAddr().c_str());
			SameLine();
			Text("port: %d", m_client.GetPort());
			SameLine();
			if (Button("Connect")) {
				xprintf("Connect clicked\n");
				int ret = m_client.Connect();
				if (m_sockwin) {
					if (ret == 0)
						m_sockwin->RenderText("Socket connected.\n");
					else
						m_sockwin->RenderText("m_xsock.Connect() failed\n");
				}
			}

			Text("Send params: ");
			SameLine();
			InputTextMultiline(" ", m_client.RequestRaw(), m_client.RequestRawSize());
			SameLine();
			if (Button("Send")) {
				std::string request = m_client.CookHttpRequest();

				// we're running in display loop (main) thread
				size_t nWritten = m_client.Send(request.c_str(), request.size());
				if (m_console) {
					if (nWritten == request.size()) {
						if (m_sockwin)
							m_sockwin->RenderText("Sent " + std::to_string(nWritten) + " bytes\n");

						// gxdq runs it's own background thread
						gxdq.Post([&]() {
							FLESS("Send wrote %d bytes\n", nWritten);

							char replyBuff[2048];
							memset(replyBuff, 0, sizeof(replyBuff));

							int nRead = m_client.Recv(replyBuff, sizeof(replyBuff));

							std::string reply = m_client.CookHttpReply(replyBuff);

							if (nRead > 0) {
								FLESS("Got %d bytes back\n", nRead);

								// lambda: xig's Animation() func runs this, ie: main thread.
								m_xdq->Post([&,reply]() {
									m_console->Clear();
									m_console->RenderText(reply);
								});
							}
						});
					}
					else
						m_console->RenderText("m_xsock.Send() failed, sent " + std::to_string(nWritten) + " bytes\n");
				}
			}
		}
		End();
	}

	bool m_show = true;
	XHttpClient m_client;
	XDispatchQueue* m_xdq;
	XGLGuiCanvas *m_sockwin = nullptr;
	XGLGuiCanvas *m_console = nullptr;
	std::thread::id m_tid;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	glm::vec3 cameraPosition(-5, -30, 5.4f);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition * -1.0f + glm::vec3(10, 0, 10));
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/diffuse", [&]() {
		shape = new XGLSphere(1.0, 32); return shape;
		shape->model = glm::translate(glm::mat4(), glm::vec3(0.0, -5.0, 0.0));
	});

	AddShape("shaders/diffuse", [&]() { xig = new ImGuiMenu(*this);	return xig;	});
}
