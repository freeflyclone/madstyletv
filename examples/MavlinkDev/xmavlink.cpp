#include "xmavlink.h"

XMavlink::ReadThread::ReadThread(XUart *px) : XThread("ReadThread"), pXUart(px) {
	Start();
}

XMavlink::ReadThread::~ReadThread() {
	Stop();
}

void XMavlink::ReadThread::Run() {
	while (IsRunning())
		if (pXUart->Read(&cp, 1))
			if ((parseState = mavlink_parse_char(MAVLINK_COMM_1, cp, &msg, &stat)) == MAVLINK_FRAMING_OK)
				MessageDump(msg);
}

bool XMavlink::ReadThread::MessageDump(mavlink_message_t msg) {
	bool retVal = false;
	switch (msg.msgid) {
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
		xprintf("msgid: %d\n", msg.msgid);
		break;
	}
	return retVal;
}

XMavlink::XMavlink(std::string portName) : XUart(portName) {
	rxThread = new ReadThread(this);
}

XMavlink::~XMavlink() {
	delete rxThread;
}
