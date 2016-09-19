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

#include <al.h>
#include <alc.h>

short audioBuffer[48000 * 4];

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	ALCdevice *audioDevice;
	ALCcontext *audioContext;

	if ((audioDevice = alcOpenDevice(NULL)) == NULL) {
		throwXGLException("alcOpenDevice() failed");
	}

	if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL) {
		throwXGLException("alcCreateContext() failed\n");
	}

	alcMakeContextCurrent(audioContext);

	alGetError();

	ALuint buffer;
	alGenBuffers(1, &buffer);
	ALenum error = alGetError();
	if (error != AL_NO_ERROR) {
		throwXGLException("alGenBuffers() failed to create a buffer");
	}

	{// build a sine wave in "audioBuffer"
		int NSAMPLES = sizeof(audioBuffer) / sizeof(audioBuffer[0]);
		for (int i = 0; i < NSAMPLES; i++) {
			double value = 2.0 * (double)i / (double)128 * M_PI;
			audioBuffer[i] = (short)(sin(value) * 32767.0);
		}
	}

	alBufferData(buffer, AL_FORMAT_MONO16, audioBuffer, sizeof(audioBuffer), 48000);
	error = alGetError();
	if (error != AL_NO_ERROR) {
		throwXGLException("alBufferData() failed");
	}

	ALuint source = 0;
	alGenSources(1, &source);
	error = alGetError();
	if (error != AL_NO_ERROR) {
		throwXGLException("alGenSources() failed");
	}

	alSourcei(source, AL_BUFFER, buffer);
	error = alGetError();
	if (error != AL_NO_ERROR) {
		throwXGLException("alSource() failed");
	}

	alSourcePlay(source);

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
		float sinFunc = sin(clock / 40.0f) * 10.0f;
		float cosFunc = cos(clock / 40.0f) * 10.0f;
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(cosFunc, sinFunc, 5.4f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		s->model = translate * rotate * scale;

		s->m.ambientColor = blue;
		s->m.diffuseColor = blue;
		s->Bind();
	};
	shape->SetTheFunk(transform);
}
