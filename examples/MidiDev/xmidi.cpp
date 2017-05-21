#include "xmidi.h"

void CALLBACK MidiInProc(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)	{
	XMidiInput *pmi = (XMidiInput*)dwInstance;
	int key;
	switch (wMsg) {
	case MIM_OPEN:
		xprintf("MidiInProc(): Open\n");
		break;

	case MIM_CLOSE:
		xprintf("MidiInProc(): Close\n");
		break;

	case MIM_DATA:
		key = LOBYTE(LOWORD(dwParam1)) << 8 | HIBYTE(LOWORD(dwParam1));
		//xprintf("MidiInProc(): Data %02X %02X %02X, (%04X) %d\n", LOBYTE(LOWORD(dwParam1)), HIBYTE(LOWORD(dwParam1)), LOBYTE(HIWORD(dwParam1)), key, dwParam2);
		pmi->KeyEvent(key, LOBYTE(HIWORD(dwParam1)));
		break;

	case MIM_LONGDATA:
		xprintf("MidiInProc(): LongData\n");
		break;

	case MIM_ERROR:
		xprintf("MidiInProc(): Error\n");
		break;

	case MIM_LONGERROR:
		xprintf("MidiInProc(): LongError\n");
		break;
	}
}


XMidiInput::XMidiInput(std::wstring devName) : deviceName(devName), XThread("XMidiInput::ReadThread") {
	xprintf("XMidiInput::XMidiInput()\n");

	unsigned int numMidiIn = midiInGetNumDevs();
	MIDIINCAPS caps;
	xprintf("Found %d MIDI devices\n", numMidiIn);

	for (unsigned int i = 0; i < numMidiIn; i++) {
		midiInGetDevCaps(i, &caps, sizeof(caps));
		deviceList[caps.szPname] = { i, caps };
	}

	xprintf("Enumerated %d devices\n", deviceList.size());
	for (auto md : deviceList) {
		xprintf("Device %S, idx: %d\n", md.second.second.szPname, md.second.first);
	}
}

XMidiInput::~XMidiInput() {
	xprintf("XMidiInput::~XMidiInput()\n");
}

void XMidiInput::Open() {
	MidiInIterator i;

	xprintf("Looking for: '%S'\n", deviceName.c_str());

	if ((i = deviceList.find(deviceName)) == deviceList.end()) {
		xprintf("no device with that name\n");
		return;
	}

	xprintf("Found it: ", i->second.second.szPname);

	if (midiInOpen(&hMidiIn, i->second.first, (DWORD_PTR)MidiInProc, (DWORD_PTR)this, CALLBACK_FUNCTION) != MMSYSERR_NOERROR) {
		xprintf("device open failed\n");
		return;
	}
}

void XMidiInput::Run() {
	Open();

	midiInStart(hMidiIn);

	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(200));
	}
}
