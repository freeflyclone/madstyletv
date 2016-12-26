#ifndef XMAVLINK_H
#define XMAVLINK_H

#include "xclasses.h"
#include "xuart.h"
#include "ardupilotmega/mavlink.h"

struct XMavlinkMessages {
	mavlink_message_t msg;
	mavlink_status_t stat;
	mavlink_attitude_t attitude;
	mavlink_sys_status_t sysStatus;
	mavlink_vfr_hud_t vfrHud;
	mavlink_statustext_t statusText;
};

class XMavlink : public XUart {
public:
	class ReadThread : public XThread {
	public:
		ReadThread(XMavlink *);
		~ReadThread();
		void Run();
		bool MessageDump();

	private:
		unsigned char cp, parseState;
		XMavlink *pMavlink;
	};

	class WriteThread : public XThread {
	public:
		WriteThread(XMavlink *);
		~WriteThread();
		void Run();
		bool WriteMessage(mavlink_message_t);

	private:
		XMavlink *pMavlink;
	};

	XMavlink(std::string portName);
	~XMavlink();

	ReadThread *rThread;
	WriteThread *wThread;

	XMavlinkMessages messages;
};


#endif