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

#ifdef WIN32
#define KKS25 L"Komplete Kontrol - 1"
#define LAUNCHPAD L"2- Launchpad S"
#define RIGKONTROL L"Rig Kontrol 3 MIDI In"
#define LPD8MK2 L"LPD8 mk2"
#else
#define KKS25 L"KOMPLETE KONTROL S25"
#define LAUNCHPAD L"Launchpad S"
#define RIGKONTROL L"Rig Kontrol 3 MIDI In"
#endif

XMidiInput *pKontrol25 = nullptr;
XGLCube *kontrol25Cube;

XMidiInput *pLaunchpad = nullptr;
XGLCube *launchpadCube;
glm::vec3 launchpadTranslate = { 10.0f, 10.0f, 0.0f };

XMidiInput *pRigKontrol;

XMidiInput *pLpd8;
XGLCube *lpd8Cube;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	if (false) {
		AddShape("shaders/specular", [&](){ kontrol25Cube = new XGLCube(); return kontrol25Cube; });
		pKontrol25 = new XMidiInput(KKS25);
		pKontrol25->AddKeyFunc(0xB00e, [this](int key, int flags) {
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(1, 1, (flags / 12.8f)));
			glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, (flags / 12.8)));
			kontrol25Cube->model = translate * scale;
		});
		try {
			pKontrol25->Start();
		}
		catch (std::runtime_error e) {
			xprintf("Open failed: %s\n", e.what());
		}
	}

	if (false) {
		AddShape("shaders/specular", [&](){ launchpadCube = new XGLCube(); return launchpadCube; });
		launchpadCube->model = glm::translate(glm::mat4(), launchpadTranslate);

		pLaunchpad = new XMidiInput(LAUNCHPAD);
		pLaunchpad->AddKeyFunc(0x9000, [this](int key, int flags) {
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(1, 1, (flags > 0) ? 2 : 1));
			launchpadCube->model = glm::translate(glm::mat4(),launchpadTranslate) * scale;
		});
		try {
			pLaunchpad->Start();
		}
		catch (std::runtime_error e) {
			xprintf("Open failed: %s\n", e.what());
		}
	}

	if (false){
		pRigKontrol = new XMidiInput(RIGKONTROL);
		pRigKontrol->AddKeyFunc({ 0x8000, 0xE000 }, [this](int key, int flags) {
			xprintf("Key: %04X: %d\n", key, flags);
		});
		try {
			pRigKontrol->Start();
		}
		catch (std::runtime_error e) {
			xprintf("Open failed: %s\n", e.what());
		}
	}

	if (true) {
		pLpd8 = new XMidiInput(LPD8MK2);
		pLpd8->AddKeyFunc({ 0x8000, 0xE000 }, [this](int key, int flags) {
			xprintf("Key: %04X: %d\n", key, flags);
		});
		try {
			pLpd8->Start();
		}
		catch (std::runtime_error e) {
			xprintf("Open failed: %s\n", e.what());
		}
	}
}
