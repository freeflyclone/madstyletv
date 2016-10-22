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
		height(h)
	{
		std::string shaderPath = pathToAssets + "/shaders/tex";
		fboQuadShader = new XGLShader(shaderPath);
		fboQuadShader->Compile(shaderPath);

		shaderPath = pathToAssets + "/shaders/fullscreen";
		fboFullscreenShader = new XGLShader(shaderPath);
		fboFullscreenShader->Compile(shaderPath);

		fboQuad = new XGLTexQuad(w, h, c);
		fboQuad->Load(fboQuadShader, fboQuad->v, fboQuad->idx);
		fboQuad->uniformLocations = fboQuadShader->materialLocations;

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-20.0, 0, 5.625f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

		fboQuad->model = translate * rotate * scale;
	};

	void Render(float clock) {
		frameBuffer.Render(std::bind(&ImageProcessing::FBRender, this));

		glProgramUniformMatrix4fv(fboQuad->shader->programId, fboQuad->shader->modelUniformLocation, 1, false, (GLfloat *)&fboQuad->model);
		GL_CHECK("glProgramUniformMatrix4fv() failed");

		fboQuad->XGLBuffer::Bind();
		fboQuad->XGLMaterial::Bind(fboQuad->shader->programId);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBuffer.intFbo);
		GL_CHECK("glBindFrameBuffer(GL_READ_FRAMEBUFFER,fb->fbo) failed");

		glReadPixels(0, 0, frameBuffer.width, frameBuffer.height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
		GL_CHECK("glReadPixels() failed");

		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameBuffer.width, frameBuffer.height, 0, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
		GL_CHECK("glTexImage() didn't work");

		fboQuad->Draw();

		XGLTexQuad::Render(clock);
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
		XGLBuffer::Bind();

		//...BUT use the fullscreen shader instead of the one
		// we got launched with.
		glUseProgram(fboFullscreenShader->programId);

		// draw the geometry.  Basically, this does a 2D fill of entire
		// FBO texture with our camera data
		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");

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
	XGLFramebuffer frameBuffer;
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
	const int camChannels = 3;


	AddShape("shaders/tex", [&](){ shape = new ImageProcessing(camWidth, camHeight, camChannels); return shape; });
	shape->windowWidth = &width;
	shape->windowHeight = &height;

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.625f));
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
