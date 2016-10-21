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
	ImageProcessing(std::string path, int w, int h, int c, GLubyte *img, bool flip) : 
		XGLTexQuad(path, w, h, c, img, flip),
		width(w),
		height(h)
	{
		std::string shaderPath = pathToAssets + "/shaders/tex";
		fboQuadShader = new XGLShader(shaderPath);
		fboQuadShader->Compile(shaderPath);

		std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
		fboQuad = new XGLTexQuad(imgPath);
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

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameBuffer.width, frameBuffer.height, 0, GL_BGR, GL_UNSIGNED_BYTE, mappedBuffer);
		GL_CHECK("glGetTexImage() didn't work");

		fboQuad->Draw();

		XGLTexQuad::Render(clock);
	}
	
	void FBRender() {
		//xprintf("FBRender()\n");
		glClearColor(0.25f, 0.5f, 0.75f, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0);
	}

	int width, height;
	XGLTexQuad *fboQuad;
	XGLShader *fboQuadShader;
	XGLFramebuffer frameBuffer;
	GLubyte mappedBuffer[1920 * 1080 * 4];
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

	AddShape("shaders/rgb2gray", [&](){ shape = new ImageProcessing(imgPath, image.cols, image.rows, image.channels(), image.data, true); return shape; });
	//shape->AddTexture(imgPath, image2.cols, image2.rows, image2.channels(), image2.data, true);

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

				glActiveTexture(GL_TEXTURE0 + activeTexture);
				GL_CHECK("glActiveTexture() failed");

				glBindTexture(GL_TEXTURE_2D, ipShape->texIds[activeTexture]);
				GL_CHECK("glBindTexture() failed");

				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pct->width, pct->height, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);
				GL_CHECK("glGetTexImage() didn't work");

				//activeTexture ^= 1;
			}
		}
	};
	shape->preRenderFunction = getCameraFrame;

	pct = new CameraThread("CameraThread");
	pct->Start();

	AddChild(pct);
}
