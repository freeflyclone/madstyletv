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

class CameraThread : public xthread {
public:
	CameraThread(std::string n) : xthread(n), readFrame(0) {};

	void Run() {
		cap.open(0);

		if (!cap.isOpened()) {
			xprintf("VideoCapture init failed\n");
			exit(-1);
		}

		cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

		while (1) {
			cap >> frame[readFrame ^= 1];
		}
	}

	cv::VideoCapture cap;
	cv::Mat frame[2];
	int readFrame;
	GLubyte imageBuffer[1920 * 1080 * 4];
};

CameraThread ct("CameraThread");

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

	// have the upright texture scaled up and made 16:9 aspect, and orbiting the origin
	// to highlight use of the callback function for animation of a shape.  Note that this function
	// runs once per frame BEFORE the shape's geomentry is rendered.  A lot can
	// be done here. Hint: scripting, physics(?)
	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		float sinFunc = sin(clock / 120.0f) * 10.0f;
		float cosFunc = cos(clock / 120.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		s->model = translate * rotate * scale;

		s->m.ambientColor = blue;
		s->m.diffuseColor = blue;
		s->b.Bind();

		//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, imageBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ct.frame[0].cols, ct.frame[0].rows, 0, GL_BGR, GL_UNSIGNED_BYTE, ct.frame[ct.readFrame].data);
		GL_CHECK("glGetTexImage() didn't work");
		//xprintf("frame: %d by %d, %d channels\n", frame.cols, frame.rows, frame.channels());
	};
	shape->SetTheFunk(transform);

	ct.Start(ct);
}
