/**************************************************************
** GUIDevBuildScene.cpp
**
** Demonstrates interaction with the GUI stack. BuildScene()
** is called after BuildGUI() by the ExampleXGL constructor,
** so it is safe to assume it exists at this point.
**
** Here is where we demonstrate adding to the GUI stack, and
** connecting GUI stack objects to world objects for a
** given application.
**************************************************************/
#include "ExampleXGL.h"
#include "xuart.h"
#include "ardupilotmega/mavlink.h"

class XMavlink : public XUart, public XThread {
public:
	XMavlink(std::string portName) : XUart(portName), XThread(portName + "Thread") {
		Start();
	}
	~XMavlink() {
		Stop();
	}

	bool MavlinkMessageDump(mavlink_message_t msg) {
		bool retVal = false;
		switch (msg.msgid) {
			case MAVLINK_MSG_ID_HEARTBEAT:
				retVal = true;
				break;

			case MAVLINK_MSG_ID_ATTITUDE:
				mavlink_msg_attitude_decode(&msg, &attitude);
				xprintf("Attitude: %0.4f, %0.4f, %0.4f\n", attitude.pitch, attitude.roll, attitude.yaw);
				retVal = true;
				break;

			case MAVLINK_MSG_ID_STATUSTEXT:
				mavlink_msg_statustext_decode(&msg, &statusText);
				xprintf("StatusText: %s\n", statusText.text);
				retVal = true;
				break;
		}
		return retVal;
	}

	void Run() {
		while (IsRunning())
			if (Read(&cp, 1))
				if ((parseState = mavlink_parse_char(MAVLINK_COMM_1, cp, &msg, &stat)) == MAVLINK_FRAMING_OK)
					MavlinkMessageDump(msg);
	}

	unsigned char cp;
	unsigned char buffer[512];
	mavlink_message_t msg;
	mavlink_status_t stat;
	mavlink_attitude_t attitude;
	mavlink_statustext_t statusText;

	unsigned char parseState;
};

class XGLGuiTextEdit : public XGLGuiCanvas {
public:
	XGLGuiTextEdit(XGL *xgl, std::string name, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h){
		SetName(name);
		attributes.ambientColor = XGLColors::black;
		attributes.ambientColor.a = 0.7f;
		attributes.diffuseColor = XGLColors::black;

		AddChildShape("shaders/ortho", [xgl, this, name, x, y]() { label = new XGLGuiLabel(xgl, name, x, y); return label; });
		label->model = glm::translate(glm::mat4(), glm::vec3(-(label->labelWidth + label->labelPadding), 0, 0.0));

		model = glm::translate(glm::mat4(), glm::vec3(x + (label->labelWidth + label->labelPadding), y, 0));
		SetMouseFunc([xgl, this](float x, float y, int flags){
			static int oldFlags = 0;

			if (flags & 1)
				CaptureMouse();
			else
				ReleaseMouse();

			// if a button has changed state...
			if (flags ^ oldFlags) {
				//... if the left button is presently down...
				if (flags & 1) {
					// ...if I don't have focus...
					if (!HasFocus()) {
						//...but someone else does, take it from them...
						if (Focused()) {
							xprintf("%s released keyboard\n", Focused()->Name().c_str());
							Focused()->ReleaseKeyboard();
						}
						// ...and then claim it for myself
						CaptureKeyboard();
						xprintf("%s captured keyboard\n", Name().c_str());
					}
				}
				oldFlags = flags;
			}

			return true;
		});
	}

private:
	XGLGuiLabel *label;
};

XMavlink *xmavlink;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLGuiManager *gm = GetGuiManager();
	XGLGuiWindow *gw;
	XGLGuiTextEdit *gte;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 480, 20, 360, 200); return gw; });
	gw->attributes.diffuseColor = { 0.7, 0.7, 0.7, 1.0 };
	gw->attributes.ambientColor = { 0.1, 0.1, 0.1, 0.5 };

	gw->SetPenPosition(10, 20);
	gw->RenderText("Container for XGLGuiTextEdit fields.\n", 20);

	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 1", 20, 40, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 2", 20, 80, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 3", 20, 120, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 4", 20, 160, 200, 24); return gte; });

	try {
		xmavlink = new XMavlink("\\\\.\\COM17");
	}
	catch (std::runtime_error e) {
		xprintf("That didn't work: %s\n", e.what());
	}
}
