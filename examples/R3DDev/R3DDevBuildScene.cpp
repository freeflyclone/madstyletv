/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"

#define __BASEFILE__ (strrchr(__FILE__, '\\') + 1)
#define FUNCENTER (xprintf("%s:%d: >> %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define FUNCEXIT  (xprintf("%s:%d: << %s()\n", __BASEFILE__, __LINE__, __FUNCTION__))
#define LOG(fmt, ...) { xprintf("%s:%d: " fmt "\n", __BASEFILE__, __LINE__, __VA_ARGS__); }

#include "XGLREDCuda.h"

class R3DPlayer : public XGLREDCuda, public XThread {
public:
	R3DPlayer(const std::string& fname) : XGLREDCuda(fname), XThread("R3DPlayerThread") {
		FUNCENTER;


		LOG("Clip resolution = %u x %u", m_width, m_height);

		FUNCEXIT;
	};

	~R3DPlayer() {
	}

	void Run() {
		FUNCENTER;

		while (IsRunning()) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
		}

		FUNCEXIT;
	}

};

R3DPlayer *player;

void ExampleXGL::BuildScene() {

	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex", [&](){ player = new R3DPlayer(r3DClipName); return player; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	player->model = translate * rotate * scale;

	player->SetAnimationFunction([&](float clock) {
		xprintf("%s\n", __FUNCTION__);
	});

	player->Start();
}
