/**************************************************************
** OpenCVTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single texture-mapped quad, along with
** scaling, translating and rotating the quad using the GLM
** functions, and doing those inside an animation callback.
**
** This is a copy of Example05, but utilizing OpenCV to load
** the image instead.
**************************************************************/
#include "ExampleXGL.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <xthread.h>
#include <xfifo.h>

#define NUM_SLOTS 8

class CameraThread : public XGLObject, public XThread {
public:
	CameraThread(std::string n) : XGLObject(n), XThread(n), slots(NUM_SLOTS), emptyCount(NUM_SLOTS), fullCount(0) {
		SetName(n);
		readIdx = writeIdx = 0;
	};

	void Run() {
		cap.open(0);

		if (!cap.isOpened()) {
			xprintf("VideoCapture init failed\n");
			exit(-1);
		}

		cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

		while (IsRunning()) {
			emptyCount.wait();
			int idx = writeIdx & (slots - 1);
			cap >> frame[idx];
			writeIdx++;
			fullCount.notify();
		}
	}

	const int slots;
	XSemaphore fullCount;
	XSemaphore emptyCount;

	int64 readIdx, writeIdx;

	cv::VideoCapture cap;
	cv::Mat frame[NUM_SLOTS];
	int readFrame;
};

CameraThread *pct;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	cv::Mat image;
	image = cv::imread(imgPath, cv::IMREAD_UNCHANGED);

	if (!image.data) {
		xprintf("imread() failed\n");
		exit(-1);
	}

	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(imgPath, image.cols, image.rows, image.channels(), image.data, true); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = rotate * scale;

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		if (clock > 0.0f) {
			float sinFunc = sin(clock / 120.0f) * 10.0f;
			float cosFunc = cos(clock / 120.0f) * 10.0f;
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
			glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
			glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			s->model = translate * rotate * scale;
		}
		s->b.Bind();

		if (pct != NULL && pct->IsRunning()) {
			/**/
			while (pct->readIdx < (pct->writeIdx-1)) {
				pct->fullCount.wait();
				int idx = pct->readIdx & (pct->slots - 1);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pct->frame[idx].cols, pct->frame[idx].rows, 0, GL_BGR, GL_UNSIGNED_BYTE, pct->frame[idx].data);
				GL_CHECK("glGetTexImage() didn't work");

				pct->readIdx++;
				pct->emptyCount.notify();
			}
			/**/
		}
	};
	shape->SetTheFunk(transform);

	pct = new CameraThread("CameraThread");

	xprintf("emptyCount: %d, fullCount: %d\n", pct->emptyCount, pct->fullCount);
	pct->Start();

	AddChild(pct);
}
