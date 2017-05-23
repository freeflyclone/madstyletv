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
#else
#define KKS25 L"KOMPLETE KONTROL S25"
#define LAUNCHPAD L"Launchpad S"
#endif

XMidiInput *pKontrol25 = nullptr;
XGLCube *kontrol25Cube;

XMidiInput *pLaunchpad = nullptr;
XGLCube *launchpadCube;
glm::vec3 launchpadTranslate = { 10.0f, 10.0f, 0.0f };

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	if (true) {
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

	if (true) {
		AddShape("shaders/specular", [&](){ launchpadCube = new XGLCube(); return launchpadCube; });
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
}
