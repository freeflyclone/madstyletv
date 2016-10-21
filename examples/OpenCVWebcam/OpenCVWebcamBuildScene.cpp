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

class ImageProcessingShape : public XGLTexQuad {
public:
	ImageProcessingShape(std::string path, int w, int h, int c, GLubyte *img, bool flip) : 
		XGLTexQuad(path, w, h, c, img, flip),
		width(w),
		height(h),
		shmem(DEFAULT_FILE_NAME)
	{
		xprintf("ImageProcessingShape()\n");
		const int SAMPLES = 8;
		
		return;

		glGenFramebuffers(1, &fbo);
		GL_CHECK("glGenFramebuffers() failed");

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		GL_CHECK("glBindFramebuffer() failed");

		glGenTextures(1, &texture);
		GL_CHECK("glGenTextures() failed");

		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
		GL_CHECK("glBindTexture() failed");

		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES, GL_RGB, width, height, GL_TRUE);
		GL_CHECK("glTexImage2D() failed");

		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
		GL_CHECK("glBindTexture(0) failed");

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texture, 0);
		GL_CHECK("glFramebufferTexture() failedn");

		// The depth buffer
		glGenRenderbuffers(1, &depth);
		GL_CHECK("glGenRenderbuffers() failed");

		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		GL_CHECK("glBindRenderbuffer() failed");

		glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES, GL_DEPTH_COMPONENT, width, height);
		GL_CHECK("glRenderbufferStorage() failed");

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
		GL_CHECK("glFramebufferRenderbuffer() failed");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		GL_CHECK("glBindFrameBuffer(0) failed");

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

		// now let's generate the intermediate FBO...
		glGenFramebuffers(1, &intFbo);
		GL_CHECK("glGenFramebuffers() failed");

		glBindFramebuffer(GL_FRAMEBUFFER, intFbo);
		GL_CHECK("glBindFramebuffer() failed");

		glGenTextures(1, &intTexture);
		GL_CHECK("glGenTextures() failed");

		glBindTexture(GL_TEXTURE_2D, intTexture);
		GL_CHECK("glBindTexture() failed");

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		GL_CHECK("glTexImage2D() failed");

		glBindTexture(GL_TEXTURE_2D, 0);
		GL_CHECK("glBindTexture(0) failed");

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, intTexture, 0);
		GL_CHECK("glFramebufferTexture() failedn");

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			xprintf("glCheckFramebufferStatus() != GL_FRAMEBUFFER_COMPLETE\n");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		GL_CHECK("glBindFrameBuffer(0) failed");

		std::string shaderName = pathToAssets + "/shaders/rgb2gray";
		imgProc = new XGLShader(shaderName);
		imgProc->Compile(shaderName);
	};

	void Render(float clock) {
		/*
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		GL_CHECK("glBindFramebuffer() failed");

		// render the geometry to the FBO here!!
		XGLBuffer::Bind();

		glUseProgram(imgProc->programId);
		GL_CHECK("glUseProgram() failed\n");

		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");

		// FBO is multi-sample for anti-aliasing.  (Do I need this?)
		// to use a a texture we have to blit to a uni-sample FBO
		// set the the "fbo" as the read for the blit
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		GL_CHECK("glBindFrameBuffer(GL_READ_FRAMEBUFFER,fb->fbo) failed");

		// set the "intFbo" for as the write for the blit
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intFbo);
		GL_CHECK("glBindFrameBuffer(GL_DRAW_FRAMEBUFFER, 0) failed");

		// do the blit to "intFbo"
		glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		GL_CHECK("glBlitFramebuffer() failed");

		glBindFramebuffer(GL_READ_FRAMEBUFFER, intFbo);
		GL_CHECK("glBindFrameBuffer(GL_READ_FRAMEBUFFER,fb->fbo) failed");

		glReadBuffer(GL_COLOR_ATTACHMENT0);
		GL_CHECK("glReadBuffer() failed");

		glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, shmem.mappedBuffer);
		GL_CHECK("glReadPixels() failed");

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		*/
		XGLTexQuad::Render(clock);
	}

	XSharedMem shmem;

	// offscreen MSAA framebuffer
	GLuint fbo;
	GLuint texture;
	GLuint depth;

	// offscreen intermediate framebuffer
	GLuint intFbo;
	GLuint intTexture;

	int width, height;

	XGLShader *imgProc;
};

class CameraThread : public XGLObject, public XThread {
public:
	CameraThread(std::string n) : XGLObject(n), XThread(n), matFifo(4) {
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
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
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
	int width, height;
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

	cv::Mat image2 = cv::Mat(image);

	AddShape("shaders/rgb2gray", [&](){ shape = new ImageProcessingShape(imgPath, image.cols, image.rows, image.channels(), image.data, true); return shape; });
	shape->AddTexture(imgPath, image2.cols, image2.rows, image2.channels(), image2.data, true);

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk getCameraFrame = [&](XGLShape *s, float clock) {
		cv::Mat img;
		static int activeTexture = 0;

		if (pct != NULL && pct->IsRunning()) {
			while (pct->matFifo.Size()) {
				img = pct->matFifo.Get();

				glActiveTexture(GL_TEXTURE0 + activeTexture);
				GL_CHECK("glActiveTexture() failed");

				glBindTexture(GL_TEXTURE_2D, s->texIds[activeTexture]);
				GL_CHECK("glBindTexture() failed");

				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pct->width, pct->height, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);
				GL_CHECK("glGetTexImage() didn't work");

				activeTexture ^= 1;
			}
		}
	};
	shape->preRenderFunction = getCameraFrame;


	pct = new CameraThread("CameraThread");
	pct->Start();

	AddChild(pct);
}
