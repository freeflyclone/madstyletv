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

class ImageProcessing : public XGLTexQuad {
public:
	ImageProcessing(int w, int h, int c) : 
		XGLTexQuad(w, h, c),
		width(w),
		height(h),
		frameBuffer(NULL)
	{
		AddTexture(width, height, c);
		frameBuffer = new XGLFramebuffer(width, height, texIds[0], texIds[1]);
		xprintf("ImageProcessing::ImageProcessing() frameBuffer\n");
	};

	// overide of XGLShape::Render(), which means if we're in this function
	// we're being called by normal shape rendering chain. Doing this to
	// add the rendering of the FBO to the chain, which is the whole reason
	// for this derived class.
	void Render(float clock) {
		if (0) {
			glProgramUniformMatrix4fv(shader->programId, shader->modelUniformLocation, 1, false, (GLfloat *)&model);
			GL_CHECK("glProgramUniformMatrix4fv() failed");
			XGLBuffer::Bind(true);
			XGLMaterial::Bind(shader->programId);
			Draw();
			Unbind();
		}
		else
			XGLShape::Render(0.0f);

		frameBuffer->Render(std::bind(&ImageProcessing::FBRender, this));
	}
	
	void FBRender() {
		// this is a 2D render of the camera quad
		// to a "fullscreen" FBO.  Disabling
		// depth testing allows for not calling
		// glClear().
		glDisable(GL_DEPTH_TEST);

		// want to render to the entire FBO bitmap texture
		glViewport(0, 0, width, height);
		GL_CHECK("glViewport() failed");

		// setup this XGLTexQuad vertex attributes & stuff...
		XGLBuffer::Bind(true);

		glUniform1i(glGetUniformLocation(shader->programId, "mode"), 1);
		GL_CHECK("glUniform1i() failed()");

		// draw the geometry.  Basically, this does a 2D fill of entire
		// FBO texture with our camera data
		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");

		glUniform1i(glGetUniformLocation(shader->programId, "mode"), 0);
		GL_CHECK("glUniform1i() failed()");

		// depending on the code that created this object to set these immediately
		// used so that we can return the glViewport to the proper window dimensions
		// after setting it above ^^
		glViewport(0, 0, *windowWidth, *windowHeight);

		glEnable(GL_DEPTH_TEST);

		XGLBuffer::Unbind();
	}

	int width, height;
	XGLTexQuad *fboQuad;
	XGLShader *fboQuadShader,*fboFullscreenShader;
	XGLFramebuffer *frameBuffer;
	GLubyte mappedBuffer[1920 * 1080 * 4];

	int *windowWidth, *windowHeight;
};

class CameraThread : public XGLObject, public XThread {
public:
	CameraThread(std::string n, int w, int h, int c) : XGLObject(n), XThread(n), matFifo(4), width(w), height(h), channels(c) {
		SetName(n);
	};

	~CameraThread() {
		xprintf("~CameraThread()\n");
		Stop();
		cap.release();
	}
	void Run() {
		cap.open(0);

		if (!cap.isOpened()) {
			xprintf("VideoCapture init failed\n");
			exit(-1);
		}

		cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
		cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
		cap.set(CV_CAP_PROP_FPS, 30.0);

		while (IsRunning()) {
			cap >> frame;
			width = frame.cols;
			height = frame.rows;
			matFifo.Put(frame);
		}
	}

	cv::VideoCapture cap;
	cv::Mat frame;

	XFifo<cv::Mat> matFifo;
	int width, height, channels;
};

CameraThread *pct;

void ExampleXGL::BuildScene() {
	ImageProcessing *shape;
	const int camWidth = 1280;
	const int camHeight = 720;
	const int camChannels = 4;

	AddShape("shaders/tex2", [&](){ shape = new ImageProcessing(camWidth, camHeight, camChannels); return shape; });
	shape->windowWidth = &width;
	shape->windowHeight = &height;

	shape->AddTexture(camWidth, camHeight, camChannels);
	shape->AddTexture(camWidth, camHeight, camChannels);
	shape->AddTexture(camWidth, camHeight, camChannels);

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(10, 0, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk getCameraFrame = [&](XGLShape *s, float clock) {
		ImageProcessing *ipShape = (ImageProcessing *)s;

		cv::Mat img;
		static int activeTexture = 0;

		if (pct != NULL && pct->IsRunning()) {
			while (pct->matFifo.Size()) {
				img = pct->matFifo.Get();

				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pct->width, pct->height, GL_BGR, GL_UNSIGNED_BYTE, img.data);
				GL_CHECK("glGetTexImage() didn't work");
			}
		}
	};
	shape->preRenderFunction = getCameraFrame;

	pct = new CameraThread("CameraThread", camWidth, camHeight, camChannels);
	pct->Start();

	AddChild(pct);
}
