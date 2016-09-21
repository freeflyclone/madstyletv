/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>

typedef struct {
	unsigned char b[960 * 810];
} ImageBuff;

ImageBuff ib;

class HighPrecisionTimer {
public:
	HighPrecisionTimer() {
		QueryPerformanceFrequency(&frequency);
	};

	unsigned int Count() {
		LARGE_INTEGER count;
		QueryPerformanceCounter(&count);
		return (count.QuadPart * (unsigned int)1000000) / frequency.QuadPart;
	}

private:
	LARGE_INTEGER frequency;
};

class AVThread : public XGLObject , public XThread {
public:
	AVThread(std::string url) : XGLObject("AVThread"), XThread("AVThread") {
		xavSrc = std::make_shared<XAVFile>(url);
		xav.AddSrc(xavSrc);
		xav.Start();
	}

	void Run() {
		while (IsRunning()) {
			unsigned char *image;
			unsigned int start = hpt.Count();
			unsigned int end = start;
			unsigned int diff = end - start;
			image = xavSrc->mVideoStream->GetBuffer();
			memcpy(&imageBuff.b, image, sizeof(imageBuff));
			ib = imageBuff;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
			do {
				end = hpt.Count();
				diff = end - start;
			} while (diff < 16667);
		}
	}

	XAV xav;
	std::shared_ptr<XAVSrc> xavSrc;
	ImageBuff imageBuff;
	HighPrecisionTimer hpt;
};

AVThread *pavt;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	std::string videoPath = pathToAssets + "/assets/CulturalPhenomenon.mp4";

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		if (pavt != NULL && pavt->IsRunning()) {
			unsigned char *image = ib.b;

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 960, 540, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)image);
			GL_CHECK("glGetTexImage() didn't work");
		}
	};
	shape->SetTheFunk(transform);

	pavt = new AVThread(videoPath);
	pavt->Start();
}
