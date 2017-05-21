#include "xmidi.h"

snd_config_t *top;

XMidiInput::XMidiInput(std::wstring devName) : deviceName(devName), XThread("XMidiInput::ReadThread") {
	int retval = snd_config_update();
	const char *id;
	if (retval < 0) {
		xprintf("snd_config_update() failed\n");
		return;
	}

	top = snd_config;

	if (top) {
		snd_config_iterator_t sci;
		for( sci = snd_config_iterator_first(top); 
			sci != snd_config_iterator_end(top); 
			sci = snd_config_iterator_next(sci)) {
			snd_config_t *entry = snd_config_iterator_entry(sci);
			snd_config_get_id(entry, &id);
			xprintf("Got one: %s\n",  id);
		}
	}

	
	// build a map of devices, by name
}

XMidiInput::~XMidiInput() {}

void XMidiInput::Open() {
	MidiInIterator i;

	xprintf("Looking for: '%S'... ", deviceName.c_str());

	if ((i = deviceList.find(deviceName)) == deviceList.end()) {
		xprintf("no device with that name\n");
		return;
	}

	xprintf("found it!\n");
	// here's where we to Linux MIDI device open
}

void XMidiInput::Run() {
	Open();

	// do OS specific start of requested MIDI Device
	while (IsRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(200));
	}
}
