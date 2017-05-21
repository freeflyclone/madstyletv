#ifndef XMIDI_H
#define XMIDI_H

#include "xinput.h"
#include "xthread.h"

#ifdef WIN32
#include <Windows.h>
#include <mmsystem.h>
// associate a device index with OS specific MIDI input device data
typedef std::pair<int, MIDIINCAPS> MidiInDevice;
#endif

typedef std::map<std::wstring, MidiInDevice> MidiInDeviceList;
typedef MidiInDeviceList::iterator MidiInIterator;

class XMidiInput : public XInput, public XThread {
public:
	XMidiInput(std::wstring devName);
	~XMidiInput();

	void Open();
	void Run();

	static void CALLBACK MidiInProc(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2);

protected:
	std::mutex mutex;
	std::wstring deviceName;
	MidiInDeviceList mdl;

#ifdef WIN32
	HMIDIIN hMidiIn;
#endif
};

#endif