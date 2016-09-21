/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>

#include <al.h>
#include <alc.h>


typedef struct {
	unsigned char b[1920 * 1620];
} ImageBuff;

ImageBuff ib;

class HighPrecisionTimer {
public:
	HighPrecisionTimer() {
		QueryPerformanceFrequency(&frequency);
	};

	long long Count() {
		LARGE_INTEGER count;
		QueryPerformanceCounter(&count);
		return (count.QuadPart * (unsigned int)1000000) / frequency.QuadPart;
	}

private:
	LARGE_INTEGER frequency;
};

class VideoStreamThread : public XThread {
public:
	VideoStreamThread(std::shared_ptr<XAVStream> s) : XThread("VideoStreamThread"), stream(s) {}

	void Run() {
		while (IsRunning()) {
			XAVBuffer image;
			long long start = hpt.Count();
			long long end = start;
			long long diff = end - start;
			image = stream->GetBuffer();
			memcpy(&imageBuff.b, image.buffer, sizeof(imageBuff));
			ib = imageBuff;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
			do {
				end = hpt.Count();
				diff = end - start;
			} while (diff < 16667);
		}
	}

	std::shared_ptr<XAVStream> stream;
	ImageBuff imageBuff;
	HighPrecisionTimer hpt;
};

class AudioStreamThread : public XThread {
public:
	AudioStreamThread(std::shared_ptr<XAVStream> s) : XThread("AudioStreamThread"), stream(s) {
		if ((audioDevice = alcOpenDevice(NULL)) == NULL) {
			throwXGLException("alcOpenDevice() failed");
		}

		if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL) {
			throwXGLException("alcCreateContext() failed\n");
		}

		alcMakeContextCurrent(audioContext);
		alGetError();

		alGenBuffers(1, &buffer);
		error = alGetError();
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

		//alSourcePlay(source);
	}

	void Run() {
		int totalBytes = 0;
		while (IsRunning()) {
			XAVBuffer audio;
			audio = stream->GetBuffer();
			totalBytes += audio.count;
		}
	}

	std::shared_ptr<XAVStream> stream;

	ALCdevice *audioDevice;
	ALCcontext *audioContext;
	ALuint buffer;
	ALenum error;
	ALuint source = 0;
	short audioBuffer[48000 * 4];
};

class AVPlayer : public XGLObject, public XThread {
public:
	AVPlayer(std::string url) : XGLObject("AVPlayer"), XThread("AVPlayerThread") {
		xavSrc = std::make_shared<XAVFile>(url);
		xav.AddSrc(xavSrc);
		xav.Start();
		vst = new VideoStreamThread(xavSrc->mVideoStream);
		ast = new AudioStreamThread(xavSrc->mAudioStream);
	}

	void Run() {
		vst->Start();
		ast->Start();

		while (IsRunning()) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(50));
		}

		vst->Stop();
		ast->Stop();
	}

	XAV xav;
	VideoStreamThread *vst;
	AudioStreamThread *ast;
	std::shared_ptr<XAVSrc> xavSrc;
};

AVPlayer *pavp;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	std::string videoPath = pathToAssets + "/assets/GOPR0541.mp4";

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		if (pavp != NULL && pavp->IsRunning()) {
			unsigned char *image = ib.b;

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1920, 1080, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)image);
			GL_CHECK("glGetTexImage() didn't work");
		}
	};
	shape->SetTheFunk(transform);

	pavp = new AVPlayer(videoPath);
	pavp->Start();
}
