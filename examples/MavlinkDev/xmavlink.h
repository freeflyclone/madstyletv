/*
** XMavlink - a class for managing an asynchronous full-duplex MAVLINK 
**   data stream over a serial port.  There is no knowledge of the
**   contents of MAVLINK packets beyond mavlink_message_t.msgid
**	 which is needed for routing.
**
** The class supports adding so-called Listener functions as needed
** (using C++ lambda functions) to provide higher level functionality,
** such as using sensor data, mission planning, vehicle configuration, etc.
*/
#ifndef XMAVLINK_H
#define XMAVLINK_H

#include "xclasses.h"

// ignore double to float precision loss warning from mavlink headers
// (rather than actually FIX it in the auto generated header files)
#pragma warning(disable:4244)
#include "ardupilotmega/mavlink.h"

class XMavlink : public XUart {
public:
	// A function type that accepts a MAVLINK message.
	typedef std::function<void(mavlink_message_t)> Listener;
	// A list of functions that allows calling multiple functions per message.
	typedef std::vector<Listener> Listeners;
	// A map of function lists, sorted by msgid, that allows calling multiple functions only for their registered msgid.
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
		bool WriteMessage(const mavlink_message_t&);

	private:
		XMavlink &pMavlink;
	};

	XMavlink(std::string portName);
	~XMavlink();
	void AddListener(uint8_t, Listener);
	void AddListener(Listener);

private:
	mavlink_message_t msg;
	mavlink_status_t stat;

	ReadThread *rThread;
	WriteThread *wThread;
	Listeners listeners;
	ListenersMap listenersMap;
};


#endif