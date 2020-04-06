/**************************************************************
** OpenCVTestBuildScene.cpp
**
** Demonstrates instantiation of a "ground"
** plane and deriving an ImageProcessing class from XGLTexQuad
** along with scaling, translating and rotating the quad using
** the GLM functions.
**
** A thread is spawned to use OpenCV to connect to a web cam
** and capture images at 30fps (if camera supports it) and
** copy those images to a simple double-buffered image buffer
** suitable for OpenGL texture upload.
**
** An animation function (runing in main rendering thread)
** uploads the previously captured image buffer from the thread
** to OpenGL texture memory.
**
**--------- The following is deprecated for now --------------
** The ImageProcessing class overrides the Render() method
** so as to inject an FBO rendering of the uploaded image,
** to allow for pixel processing with a fragment shader.
**
** Image processing is accomplished with a shader that
** reads pixels from the input texture, does some stuff, and
** writes to an output texture during the FBO render pass,
** then uses the output texture during the normal pass to
** display the results.  This shader has a "mode" uniform
** variable that is used to tell it which pass it's operating
** for.
**************************************************************/
#include "ExampleXGL.h"
#include <xthread.h>
#include "xlog.h"

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

XLOG_DEFINE("ocvWebcam", XLDebug);

using namespace cv;

class XGLWebcam : public XGLTexQuad, public XThread
{
public:
	XGLWebcam(std::string n, int idx = 0, int w = 640, int h = 360, int c = 3) 
		: XThread("XGLWebcamThread"), 
		XGLTexQuad(w,h,c),
		cameraIndex(idx),
		width(w), 
		height(h), 
		channels(c), 
		frameNumber(0)
	{
		SetName(n);

		SetAnimationFunction([&](float clock) {
			if (IsRunning() && (frameNumber > 3)) {
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texIds[0]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, videoFrame[(frameNumber - 1) & 3]);
				GL_CHECK("glGetTexImage() didn't work");
			}
		});
		XLOG(XLTrace);
	};

	~XGLWebcam() {
		Stop();
		cap.release();
	}

	void Run() {
		XLOG(XLTrace, "-->");

		cap.open(cameraIndex, CAP_DSHOW);

		if (!cap.isOpened()) {
			XLOG(XLDebug, "VideoCapture init failed\n");
			exit(-1);
		}

		auto backEnd = cap.getBackendName();

		// this is hard-coded for Logitech C920 web cam. Also works
		// on a Macbook Pro with internal camera.  May work with others.
		cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('Y', 'U', 'Y', '2'));
		cap.set(CAP_PROP_FRAME_WIDTH, width);
		cap.set(CAP_PROP_FRAME_HEIGHT, height);
		cap.set(CAP_PROP_FPS, 30.0);

		cap >> frame;

		while (IsRunning()) {
			cap >> frame;
			memcpy(videoFrame[frameNumber & 3], frame.data, (frame.cols*frame.rows*frame.channels()));
			frameNumber++;
			width = frame.cols;
			height = frame.rows;
		}
		XLOG(XLTrace, "<--");
	}

	VideoCapture cap;
	Mat frame;

	int width, height, channels;
	int cameraIndex{ 0 };
	unsigned int frameNumber;

	// ultra-simple quadruple-buffered intermediate frames from the camera
	// (ping-ponged by frameNumber&3) TODO: size this programmatically.
	GLubyte videoFrame[4][1920 * 1080 * 4];
};

void ExampleXGL::BuildScene() {
	XGLWebcam *webcam0;
	XGLWebcam *webcam1;
	XGLWebcam *webcam2;

	AddShape("shaders/tex", [&](){ webcam0 = new XGLWebcam("Camera 1", 0); return webcam0; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-10, 0, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam0->model = translate * rotate * scale;

	AddShape("shaders/tex", [&]() { webcam1 = new XGLWebcam("Camera 2", 1); return webcam1; });
	scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(10, 0, 5.625f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam1->model = translate * rotate * scale;

	AddShape("shaders/tex", [&]() { webcam2 = new XGLWebcam("Camera 2", 2); return webcam2; });
	scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(-30, 0, 5.625f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam2->model = translate * rotate * scale;

	webcam0->Start();
	webcam1->Start();
	webcam2->Start();
}
