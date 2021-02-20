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

class XGLWebcam : public VideoCapture,  public XGLTexQuad, public XThread
{
public:
	XGLWebcam(XGL* pXGL, std::string n, int idx = 0, int w = 640, int h = 360, int c = 3) 
		: XThread("XGLWebcamThread"), 
		XGLTexQuad(w,h,c),
		cameraIndex(idx),
		width(w), 
		height(h), 
		channels(c),
		pXGL(pXGL),
		frameNumber(0)
	{
		SetName(n);

		SetAnimationFunction([&](float clock) {
			if (IsRunning() && (frameNumber > 3)) {
				glActiveTexture(GL_TEXTURE0);
				GL_CHECK("glActiveTexture() didn't work");

				glBindTexture(GL_TEXTURE_2D, texIds[0]);
				GL_CHECK("glBindTexture() didn't work");

				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, videoFrame[(frameNumber - 1) & 3]);
				GL_CHECK("glGetTexImage() didn't work");
			}
		});

		XLOG(XLTrace);
	};

	~XGLWebcam() {
		Stop();
		release();
	}

	void Run() {
		XLOG(XLTrace, "-->");

		open(cameraIndex, CAP_DSHOW);

		if (!isOpened()) {
			XLOG(XLDebug, "VideoCapture init failed\n");
			exit(-1);
		}

		auto backEnd = getBackendName();

		// this is hard-coded for Logitech C920 web cam. Also works
		// Note: the following is not presently true: on a Macbook Pro with internal camera.  May work with others.
		set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
		set(CAP_PROP_FRAME_WIDTH, width);
		set(CAP_PROP_FRAME_HEIGHT, height);
		set(CAP_PROP_FPS, 30.0);

		set(CAP_PROP_BRIGHTNESS, (double)brightness);
		set(CAP_PROP_CONTRAST, (double)contrast);
		set(CAP_PROP_SATURATION, (double)saturation);
		set(CAP_PROP_HUE, (double)hue);
		set(CAP_PROP_EXPOSURE, (double)exposure);

		// this sort of works, but all 3 cameras share the same settings.  WTF?
		pXGL->menuFunctions.push_back(([this]() {
			if (ImGui::Begin("WebCam Controls", &guiWindow))
			{
				if (ImGui::SliderFloat("Brightness", &brightness, 0.0f, 256.0f, "%0.3f", 1))
					set(CAP_PROP_BRIGHTNESS, (double)brightness);

				if (ImGui::SliderFloat("Contrast", &contrast, 0.0f, 256.0f, "%0.3f", 1))
					set(CAP_PROP_CONTRAST, (double)contrast);

				if (ImGui::SliderFloat("saturation", &saturation, 0.0f, 256.0f, "%0.3f", 1))
					set(CAP_PROP_SATURATION, (double)saturation);

				if (ImGui::SliderFloat("hue", &hue, 0.0f, 256.0f, "%0.3f", 1))
					set(CAP_PROP_HUE, (double)hue);

				if (ImGui::SliderFloat("Exposure", &exposure, -10.0f, -2.0f, "%0.0f", 1))
					set(CAP_PROP_EXPOSURE, (double)exposure);
			}
			ImGui::End();
		}));

		read(frame);

		while (IsRunning()) {
			read(frame);
			memcpy(videoFrame[frameNumber & 3], frame.data, (frame.cols*frame.rows*frame.channels()));
			frameNumber++;
			width = frame.cols;
			height = frame.rows;
		}
		XLOG(XLTrace, "<--");
	}

	Mat frame;

	int width, height, channels;
	int cameraIndex{ 0 };
	unsigned int frameNumber;

	// ultra-simple quadruple-buffered intermediate frames from the camera
	// (ping-ponged by frameNumber&3) TODO: size this programmatically.
	GLubyte videoFrame[4][1920 * 1080 * 4];

private:
	bool guiWindow = true;
	float brightness{ 128 };
	float contrast{ 128 };
	float saturation{ 128 };
	float hue{ 128 };
	float exposure{ -5 };

	XGL* pXGL;
};


XGLWebcam *webcam0{ nullptr };
XGLWebcam *webcam1{ nullptr };
XGLWebcam *webcam2{ nullptr };

void ExampleXGL::BuildScene() {

	AddShape("shaders/tex", [&](){ webcam0 = new XGLWebcam(this, "Camera 1", 0); return webcam0; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-10, 0, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam0->model = translate * rotate * scale;
	/*
	AddShape("shaders/tex", [&]() { webcam1 = new XGLWebcam(this, "Camera 2", 1); return webcam1; });
	scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(10, 0, 5.625f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam1->model = translate * rotate * scale;

	AddShape("shaders/tex", [&]() { webcam2 = new XGLWebcam(this, "Camera 2", 2); return webcam2; });
	scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	translate = glm::translate(glm::mat4(), glm::vec3(-30, 0, 5.625f));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	webcam2->model = translate * rotate * scale;
	*/

	webcam0->Start();
	//webcam1->Start();
	//webcam2->Start();
}
