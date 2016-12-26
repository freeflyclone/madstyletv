#ifndef XMAVLINK_H
#define XMAVLINK_H

#include "xclasses.h"
#include "xuart.h"
#include "ardupilotmega/mavlink.h"

class XMavlink : public XUart {
public:
	class ReadThread : public XThread {
	public:
		ReadThread(XUart *px);
		~ReadThread();
		void Run();
		bool MessageDump(mavlink_message_t msg);

	private:
		unsigned char cp, parseState;
		XUart *pXUart;
		mavlink_message_t msg;
		mavlink_status_t stat;
	};

	XMavlink(std::string portName);
	~XMavlink();

	ReadThread *rxThread;

	unsigned char cp;
	unsigned char buffer[512];
	mavlink_message_t msg;
	mavlink_status_t stat;
	mavlink_attitude_t attitude;
	mavlink_sys_status_t sysStatus;
	mavlink_vfr_hud_t vfrHud;
	mavlink_statustext_t statusText;

	unsigned char parseState;

};


#endif