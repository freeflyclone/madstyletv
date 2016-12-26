#include "xmavlink.h"

// --------------------------
// XMavlink::ReadThread class
// --------------------------
XMavlink::ReadThread::ReadThread(XMavlink &pm) : XThread("XMavlink::ReadThread"), pMavlink(pm) {
	Start();
}

XMavlink::ReadThread::~ReadThread() {
	Stop();
}

// read a byte, pass it to mavlink_parse_char, if we get a message, 
// call possibly mulitple Listener functions connected to the received msgid 
// and/or possibly multiple Listener functions that receive ALL msgid's
void XMavlink::ReadThread::Run() {
	while (IsRunning())
		if (pMavlink.Read(&cp, 1))
			if ((parseState = mavlink_parse_char(MAVLINK_COMM_1, cp, &pMavlink.msg, &pMavlink.stat)) == MAVLINK_FRAMING_OK) {
				for (auto fn : pMavlink.listenersMap[pMavlink.msg.msgid])
					fn(pMavlink.msg);
				for (auto fn : pMavlink.listeners)
					fn(pMavlink.msg);
			}
}

// ---------------------------
// XMavlink::WriteThread class
// ---------------------------
XMavlink::WriteThread::WriteThread(XMavlink &pm) : XThread("XMavlink::WriteThread"), pMavlink(pm) {
	Start();
}

XMavlink::WriteThread::~WriteThread() {
	Stop();
}

void XMavlink::WriteThread::Run() {
	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
	}
}

// ---------------------------
// XMavlink main class
// ---------------------------
XMavlink::XMavlink(std::string portName) : XUart(portName), rThread(NULL), wThread(NULL) {
	rThread = new ReadThread(*this);
	wThread = new WriteThread(*this);
}

XMavlink::~XMavlink() {
	if (rThread)
		delete rThread;
	if (wThread)
		delete wThread;
}

// attach a Listener to a specific msgid
void XMavlink::AddListener(uint8_t msgid, Listener fn) {
	listenersMap[msgid].push_back(fn);
}

// attach a Listener for ALL msgids
void XMavlink::AddListener(Listener fn) {
	listeners.push_back(fn);
}
