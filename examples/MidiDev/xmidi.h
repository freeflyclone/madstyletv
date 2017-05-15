#ifndef XMIDI_H
#define XMIDI_H

#include "xinput.h"
#include "xthread.h"

class XMidiInput : public XInput, public XThread {
public:
	XMidiInput();
	~XMidiInput();

	void Run();

private:
	std::mutex mutex;
};

#endif