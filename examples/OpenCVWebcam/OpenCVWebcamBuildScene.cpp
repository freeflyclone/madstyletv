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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <xthread.h>
#include <xfifo.h>

class ImageProcessing : public XGLTexQuad {
public:
	ImageProcessing(int w, int h, int c) : 
		XGLTexQuad(),
		width(w),
		height(h),
		frameBufferObject(NULL)
	{
		// we start off with no textures in our base class constructor
		// so add the 4 we'll need now.
		AddTexture(width, height, c);
		AddTexture(width, height, c);
		AddTexture(width, height, c);
		AddTexture(width, height, c);

		frameBufferObject = new XGLFramebuffer(width, height, texIds.data(), texIds.size());

		std::string shaderName = pathToAssets + "/shaders/imageproc";
		imgShader = new XGLShader(shaderName);
		imgShader->Compile(shaderName);

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(10, 0, 5.625f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

		MakeSubQuad(3, translate * rotate * scale);

		translate = glm::translate(glm::mat4(), glm::vec3(-10, 0, 3*5.625f));
		MakeSubQuad(0, translate * rotate * scale);

		translate = glm::translate(glm::mat4(), glm::vec3(10, 0, 3 * 5.625f));
		MakeSubQuad(1, translate * rotate * scale);
	};

	// overide of XGLShape::Render(), which means if we're in this function
	// we're being called by normal shape rendering chain. Doing this to
	// add the rendering to the FBO to the chain, which is the whole reason
	// for this derived class.
	void Render(float clock) {
		frameBufferObject->Render(std::bind(&ImageProcessing::FBORender, this));

		// These ensure that the textures rendered in the FBO pass are available 
		// in the main rendering pass.  Is probably sticky in the program object, (shader)
		// and could be set in this object's constructor, if "shader" were valid at that point.
		// I'll need to refactor to make that so.
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);
		glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit3"), 3);

		XGLShape::Render(0.0);
		for (imgQuadsIterator = imgQuads.begin(); imgQuadsIterator != imgQuads.end(); imgQuadsIterator++) {
			(*imgQuadsIterator)->Render(0.0);
		}
	}
	
	void FBORender() {
		glDisable(GL_DEPTH_TEST);
		glViewport(0, 0, width, height);

		XGLBuffer::Bind();

		// These ensure that the FBO pass has access to all the GL_COLORATTACHMENT buffers
		// that have been setup in the FBO;  (Might be sticky in the FBO and not 
		// need setting per render pass)
		glBindFragDataLocation(shader->programId, 0, "color0");
		glBindFragDataLocation(shader->programId, 1, "color1");
		glBindFragDataLocation(shader->programId, 2, "color2");
		glBindFragDataLocation(shader->programId, 3, "color3");

		glUniform1i(glGetUniformLocation(shader->programId, "mode"), 1);
		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		glUniform1i(glGetUniformLocation(shader->programId, "mode"), 0);
		glViewport(0, 0, *windowWidth, *windowHeight);
		glEnable(GL_DEPTH_TEST);
		GL_CHECK("there was a problem in the FBO rendering");

		XGLBuffer::Unbind();
	}

	void MakeSubQuad(GLuint texId, glm::mat4 model) {
		XGLTexQuad *quad = new XGLTexQuad();
		quad = new XGLTexQuad();
		quad->texIds.push_back(texIds[texId]);
		quad->numTextures = 1;
		quad->Load(imgShader, quad->v, quad->idx);
		quad->uniformLocations = imgShader->materialLocations;
		quad->model = model;
		imgQuads.push_back(quad);
	}

	int width, height;
	XGLFramebuffer *frameBufferObject;
	XGLShapeList imgQuads;
	XGLShapeList::iterator imgQuadsIterator;
	XGLShader *imgShader;
	int *windowWidth, *windowHeight;
};

class CameraThread : public XGLObject, public XThread {
public:
	CameraThread(std::string n, int w, int h, int c) : XGLObject(n), XThread(n), width(w), height(h), channels(c), frameNumber(0) {
		SetName(n);
	};

	~CameraThread() {
		Stop();
		cap.release();
	}
	void Run() {
		cap.open(0);

		if (!cap.isOpened()) {
			xprintf("VideoCapture init failed\n");
			exit(-1);
		}

		// this is hard-coded for Logitech C920 web cam. Also works
		// on a Macbook Pro with internal camera.  May work with others.
		cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
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
	}

	cv::VideoCapture cap;
	cv::Mat frame;

	int width, height, channels;
	unsigned int frameNumber;

	// ultra-simple quadruple-buffered intermediate frames from the camera
	// (ping-ponged by frameNumber&3) TODO: size this programmatically.
	GLubyte videoFrame[4][1920 * 1080 * 4];
};

CameraThread *pct;

void ExampleXGL::BuildScene() {
	ImageProcessing *shape;
	const int camWidth = 1920;
	const int camHeight = 1080;
	const int camChannels = 3;

	AddShape("shaders/imageproc", [&](){ shape = new ImageProcessing(camWidth, camHeight, camChannels); return shape; });
	shape->windowWidth = &width;
	shape->windowHeight = &height;

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-10, 0, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	// animation function to grab a web cam frame from the web cam capture thread and upload it to texture memory
	XGLShape::AnimaFunk getCameraFrame = [&](XGLShape *s, float clock) {
		ImageProcessing *ipShape = (ImageProcessing *)s;

		if (pct != NULL && pct->IsRunning() && (pct->frameNumber>3) ) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, ipShape->texIds[0]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pct->width, pct->height, GL_BGR, GL_UNSIGNED_BYTE, pct->videoFrame[(pct->frameNumber-1)&3]);
			GL_CHECK("glGetTexImage() didn't work");

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, ipShape->texIds[1]);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pct->width, pct->height, GL_BGR, GL_UNSIGNED_BYTE, pct->videoFrame[(pct->frameNumber-2)&3]);
			GL_CHECK("glGetTexImage() didn't work");
		}
	};
	shape->SetTheFunk(getCameraFrame);

	pct = new CameraThread("CameraThread", camWidth, camHeight, camChannels);
	pct->Start();

	AddChild(pct);
}
