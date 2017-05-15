#include "xmidi.h"

XMidiInput::XMidiInput() : XThread("XMidiInput::ReadThread") {
	xprintf("XMidiInput::XMidiInput()\n");
}

XMidiInput::~XMidiInput() {
	xprintf("XMidiInput::~XMidiInput()\n");
}

void XMidiInput::Run() {
}
