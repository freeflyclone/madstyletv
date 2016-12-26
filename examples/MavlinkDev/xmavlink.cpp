#include "xmavlink.h"

// --------------------------
// XMavlink::ReadThread class
// --------------------------
XMavlink::ReadThread::ReadThread(XMavlink *pm) : XThread("XMavlink::ReadThread"), pMavlink(pm) {
	Start();
}

XMavlink::ReadThread::~ReadThread() {
	Stop();
}

void XMavlink::ReadThread::Run() {
	while (IsRunning())
		if (pMavlink->Read(&cp, 1))
			if ((parseState = mavlink_parse_char(MAVLINK_COMM_1, cp, &pMavlink->messages.msg, &pMavlink->messages.stat)) == MAVLINK_FRAMING_OK)
				MessageDump();
}

bool XMavlink::ReadThread::MessageDump() {
	bool retVal = false;
	switch (pMavlink->messages.msg.msgid) {
	case MAVLINK_MSG_ID_HEARTBEAT:
		retVal = true;
		break;

	case MAVLINK_MSG_ID_ATTITUDE:
		xprintf("Attitude\n");
		retVal = true;
		break;

	case MAVLINK_MSG_ID_STATUSTEXT:
		xprintf("StatusText\n");
		retVal = true;
		break;

	case MAVLINK_MSG_ID_SYS_STATUS:
		xprintf("Status\n");
		break;

	case MAVLINK_MSG_ID_VFR_HUD:
		xprintf("VFR HUD\n");
		break;

	default:
		xprintf("msgid: %d\n", pMavlink->messages.msg.msgid);
		break;
	}
	return retVal;
}

// ---------------------------
// XMavlink::WriteThread class
// ---------------------------
XMavlink::WriteThread::WriteThread(XMavlink *pm) : XThread("XMavlink::WriteThread"), pMavlink(pm) {
	Start();
}

XMavlink::WriteThread::~WriteThread() {
	Stop();
}

void XMavlink::WriteThread::Run() {
	xprintf("XMavlink::WriteThread::Run() started.\n");
	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
	}
}

// ---------------------------
// XMavlink main class
// ---------------------------
XMavlink::XMavlink(std::string portName) : XUart(portName), rThread(NULL), wThread(NULL) {
	rThread = new ReadThread(this);
	wThread = new WriteThread(this);
}

XMavlink::~XMavlink() {
	if (rThread)
		delete rThread;
	if (wThread)
		delete wThread;
}
