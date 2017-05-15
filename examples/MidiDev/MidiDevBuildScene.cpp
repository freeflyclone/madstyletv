/**************************************************************
** MidiDevBuildScene.cpp
**
** MIDI Input interface example.  Demonstrates
** a custom Input device that can trigger events in the virtual
** world according to MIDI Events received from a connected
** device.
**
** With default camera manipulation via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

#include "xmidi.h"
#include <mmsystem.h>

class MidiIn : public XMidiInput {
public:
	typedef std::pair<int, MIDIINCAPS> MidiInDevice;
	typedef std::map<std::wstring,MidiInDevice> MidiInDeviceList;
	typedef MidiInDeviceList::iterator MidiInIterator;

	MidiIn() {
		unsigned int numMidiIn = midiInGetNumDevs();
		MIDIINCAPS caps;
		xprintf("Found %d MIDI devices\n", numMidiIn);

		for (unsigned int i = 0; i < numMidiIn; i++) {
			midiInGetDevCaps(i, &caps, sizeof(caps));
			mdl[caps.szPname] = { i, caps };
		}

		xprintf("Enumerated %d devices\n", mdl.size());
		for (auto md : mdl) {
			xprintf("Device %S, idx: %d\n", md.second.second.szPname, md.second.first);
		}
	};
	~MidiIn() {};

	void Open(std::wstring deviceName) {
		MidiInIterator i;
		
		if ((i = mdl.find(deviceName)) == mdl.end())
			throw std::runtime_error("no device with that name");

		if (midiInOpen(&hMidiIn, i->second.first, (DWORD_PTR)MidiInProc, (DWORD_PTR)this, CALLBACK_FUNCTION) != MMSYSERR_NOERROR)
			throw std::runtime_error("device open failed");
	}

	static void CALLBACK MidiInProc(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
		MidiIn *pmi = (MidiIn*)dwInstance;
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

	void Run() {
		Open(L"Launchpad S");
		midiInStart(hMidiIn);

		while (IsRunning()) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(200));
		}
	}

private:
	MidiInDeviceList mdl;
	HMIDIIN	hMidiIn;
};

MidiIn *pmi = NULL;
XGLCube *cube;
void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
	AddShape("shaders/specular", [&](){ cube = new XGLCube(); return cube; });

	pmi = new MidiIn();

	pmi->AddKeyFunc({ 0x9000, 0x9000 }, [this](int key, int flags) {
	});
	pmi->AddKeyFunc({ 0xB068, 0xB06f }, [this](int key, int flags) {
		xprintf("CCHit: %04X, %04X!\n", key & 0xFF, flags);
	});
	pmi->AddKeyFunc(0x9000, [&](int key, int flags) {
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(1, 1, (flags > 0) ? 2 : 1));
		cube->model = scale;
	});

	try {
		pmi->Start();
	}
	catch (std::runtime_error e) {
		xprintf("Open failed: %s\n", e.what());
	}

}
