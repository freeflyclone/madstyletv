/**************************************************************
** R3DDevBuildScene.cpp
**
** Mission: Integrate R3DSDK fully, including CUDA
**************************************************************/
#include "ExampleXGL.h"

#include "XGLREDCuda.h"

class R3DPlayer : public XGLREDCuda {
public:
	R3DPlayer(const std::string& fname) : XGLREDCuda(fname) {
		AddCompletionFunction([this](XGLREDCuda *pRedCuda) {
			StartVideoDecode(gpuDone);
		});
	}
	~R3DPlayer() {}
};

R3DPlayer *player;
int frameNum = 0;

void ExampleXGL::BuildScene() {
	std::string r3DClipName = config.WideToBytes(config.Find(L"R3DFile")->AsString());

	AddShape("shaders/tex", [&](){ player = new R3DPlayer(r3DClipName); return player; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(17.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	player->model = translate * rotate * scale;

	player->StartVideoDecode(frameNum);
}
