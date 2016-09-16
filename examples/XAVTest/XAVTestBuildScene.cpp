/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xavfile.h>

class VideoFileThread : public XGLObject {
public:
	VideoFileThread(std::string url) : XGLObject("VideoFileThread") {
		// first thing MUST be av_register_all()
		av_register_all();
		xavFile = new XAVFile(url);
		xavFile->Start();
	}

	XAVFile *xavFile;
};

VideoFileThread *pvft;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	std::string videoPath = pathToAssets + "/assets/CulturalPhenomenon.mp4";

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	pvft = new VideoFileThread(videoPath);

	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		s->b.Bind();
		unsigned char *image;
		image = pvft->xavFile->mVideoStream->GetBuffer();

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 960, 540, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)image);
		GL_CHECK("glGetTexImage() didn't work");
	};
	shape->SetTheFunk(transform);
}
