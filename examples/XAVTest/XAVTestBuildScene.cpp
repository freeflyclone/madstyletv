/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xavfile.h>

class VideoFileThread : public XGLObject, public XThread {
public:
	VideoFileThread(std::string url) : XGLObject("VideoFileThread"), XThread("VideoFileThread") {
		// first thing MUST be av_register_all()
		av_register_all();
		xavFile = new XAVFile(url);
	}

	void Run() {
		xavFile->Start();

		while (IsRunning()) {
			xprintf("VideoFileThread::Run()\n");
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(500));
		}
	}

	XAVFile *xavFile;
};

VideoFileThread *pvft;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	std::string videoPath = pathToAssets + "/assets/CulturalPhenomenon.mp4";

	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	pvft = new VideoFileThread(videoPath);
	pvft->Start();
}
