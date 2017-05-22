#include "xmidi.h"
snd_config_t *top;
std::wstring CharToWide(char *s) {
	std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> cv;
	return cv.from_bytes(s);
}

XMidiInput::XMidiInput(std::wstring devName) : deviceName(devName), XThread("XMidiInput::ReadThread") {
	int retval = snd_config_update();
	void **hints;

	if (retval < 0) {
		xprintf("snd_config_update() failed\n");
		return;
	}

	// build a map of devices, by name
	if (snd_device_name_hint(-1, "rawmidi", &hints) < 0) {
		xprintf("query devices failed\n");
		return;
	}

	for( int i = 0; (*hints); hints++, i++) {
		char *name = snd_device_name_get_hint(*hints, "NAME");
		char *desc = snd_device_name_get_hint(*hints, "DESC");
		if (desc) {
			char *descToken = strtok(desc, ",");

			std::wstring devName = CharToWide(descToken);
			deviceList[devName] = { i, std::string(name) };
		}
	}
}

XMidiInput::~XMidiInput() {}

void XMidiInput::Open() {
	MidiInIterator i;
	int status;

	if ((i = deviceList.find(deviceName)) == deviceList.end()) {
		xprintf("no device with that name\n");
		return;
	}

	if ((status = snd_rawmidi_open(&hMidiIn, NULL, i->second.second.c_str(), 0)) < 0) {
		xprintf("snd_rawmidi_open() failed\n");
	}
}

void XMidiInput::Run() {
	int status, key,flags;
	char buffer[4];

	Open();

	while (IsRunning()) {
		if ((status = snd_rawmidi_read(hMidiIn, buffer, sizeof(buffer))) < 0) {
			xprintf("snd_rawmidi_read() returned: %s\n", status);
			continue;
		}

		key = (buffer[0]&0xFF) << 8 | (buffer[1]&0xFF);
		flags = (buffer[2]&0xFF);

		KeyEvent(key, flags);
	}
}
