#ifndef XMAVLINK_H
#define XMAVLINK_H

#include "xclasses.h"
#include "xuart.h"
#include "ardupilotmega/mavlink.h"

class XMavlink : public XUart {
public:
	// define a function type that takes a MAVLINK message
	typedef std::function<void(mavlink_message_t)> Listener;
	// define a list of those functions so we can call multiple functions per msgid
	typedef std::vector<Listener> Listeners;
	// define a map of those lists by msgid so that a function will only be called with it's registered msgid
	typedef std::map<uint8_t, Listeners> ListenersMap;

	class ReadThread : public XThread {
	public:
		ReadThread(XMavlink &);
		~ReadThread();
		void Run();

	private:
		unsigned char cp, parseState;
		XMavlink &pMavlink;
	};

	class WriteThread : public XThread {
	public:
		WriteThread(XMavlink &);
		~WriteThread();
		void Run();
		bool WriteMessage(mavlink_message_t);

	private:
		XMavlink &pMavlink;
	};

	XMavlink(std::string portName);
	~XMavlink();
	void AddListener(uint8_t, Listener);

private:
	mavlink_message_t msg;
	mavlink_status_t stat;

	ReadThread *rThread;
	WriteThread *wThread;
	ListenersMap listeners;
};


#endif