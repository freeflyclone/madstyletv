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
#include "xlog.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <xthread.h>

XLOG_DEFINE("ocvWebcam", XLDebug);

class CameraThread : public XObject, public XThread {
public:
	CameraThread(std::string n, int w, int h, int c) : XObject(n), XThread(n), width(w), height(h), channels(c), frameNumber(0) {
		SetName(n);
		XLOG(XLTrace);
	};

	~CameraThread() {
		Stop();
		cap.release();
	}
	void Run() {
		XLOG(XLTrace, "-->");

		cap.open(0);

		if (!cap.isOpened()) {
			XLOG(XLDebug, "VideoCapture init failed\n");
			exit(-1);
		}

		// this is hard-coded for Logitech C920 web cam. Also works
		// on a Macbook Pro with internal camera.  May work with others.
		cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('Y', 'U', 'Y', '2'));
		cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
		cap.set(CV_CAP_PROP_FPS, 30.0);

		while (IsRunning()) {
			cap >> frame;
			memcpy(videoFrame[frameNumber & 3], frame.data, (frame.cols*frame.rows*frame.channels()));
			frameNumber++;
			width = frame.cols;
			height = frame.rows;
		}
		XLOG(XLTrace, "<--");
	}

	cv::VideoCapture cap;
	cv::Mat frame;

	int width, height, channels;
	unsigned int frameNumber;

	// ultra-simple quadruple-buffered intermediate frames from the camera
	// (ping-ponged by frameNumber&3) TODO: size this programmatically.
	GLubyte videoFrame[4][1920 * 1080 * 4];
};

void ExampleXGL::BuildScene() {
	XGLTexQuad *shape;
	const int camWidth = 640;
	const int camHeight = 360;
	const int camChannels = 3;

	CameraThread *pct = new CameraThread("CameraThread", camWidth, camHeight, camChannels);

	AddShape("shaders/tex", [&](){ shape = new XGLTexQuad(camWidth, camHeight, camChannels); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-10, 0, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	// animation function to grab a web cam frame from the web cam capture thread and upload it to texture memory
	shape->SetAnimationFunction([pct,shape](float clock) {
		XGLTexQuad *ipShape = (XGLTexQuad *)shape;
		if (pct != NULL && pct->IsRunning() && (pct->frameNumber>3) ) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, ipShape->texIds[0]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pct->width, pct->height, GL_BGR, GL_UNSIGNED_BYTE, pct->videoFrame[(pct->frameNumber-1)&3]);
			GL_CHECK("glGetTexImage() didn't work");
		}
	});

	// Attaching a non XGL XObject to an XGLShape is accomplished as follows.
	// Doing this attaches CameraThread object to the XGL object that renders it's output.
	shape->XObject::AddChild(pct);

	pct->Start();

}
