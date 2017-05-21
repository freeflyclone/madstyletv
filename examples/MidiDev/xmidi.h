#ifndef XMIDI_H
#define XMIDI_H

#include "xinput.h"
#include "xthread.h"

#ifdef WIN32
	#include <Windows.h>
	#include <mmsystem.h>
	// associate a device index with OS specific MIDI input device data
	typedef std::pair<int, MIDIINCAPS> XMidiInDeviceProps;
	typedef HMIDIIN XMIDIDEVHANDLE;
#else
	typedef int XMIDIDEVHANDLE;
#endif

typedef std::map<std::wstring, XMidiInDeviceProps> XMidiInDeviceList;
typedef XMidiInDeviceList::iterator MidiInIterator;

class XMidiInput : public XInput, public XThread {
public:
	XMidiInput(std::wstring devName);
	~XMidiInput();

	void Open();
	void Run();
protected:
	std::mutex mutex;
	std::wstring deviceName;
	XMidiInDeviceList deviceList;

#ifdef WIN32
	XMIDIDEVHANDLE hMidiIn;
#endif
};

#endif