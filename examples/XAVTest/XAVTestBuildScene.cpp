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
#include <alext.h>
#include <iostream>

#define AUDIO_SAMPLES 1024
#define AL_CHECK(what) CheckError(__FILE__,__LINE__,what)

typedef struct {
	unsigned char b[1920 * 1620];
} ImageBuff;

ImageBuff ib;

typedef struct {
	float left;
} AudioSampleFloat;

typedef struct {
	short left;
	short right;
} AudioSampleShort;

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
		xprintf("AudioCodec name: %s\n", stream->pCodec->name);
		int sampleSize = stream->formatSize * stream->channels;

		sampleRate = stream->sampleRate;

		if ((audioDevice = alcOpenDevice(NULL)) == NULL) {
			throwXGLException("alcOpenDevice() failed");
		}

		xprintf("ALC_EXTENSIONS: '%s'\n", alcGetString(audioDevice, ALC_EXTENSIONS));
		if (alIsExtensionPresent("AL_EXT_FLOAT32"))
			xprintf("AL_EXT_FLOAT32 is present\n");

		if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL) {
			throwXGLException("alcCreateContext() failed\n");
		}

		alcMakeContextCurrent(audioContext);
		alGetError();

		alGenBuffers(XAV_NUM_FRAMES, bufferIDs);
		AL_CHECK("alGenBuffers() failed");

		for (int i = 0; i < XAV_NUM_FRAMES; i++)
			xprintf("bufferIDs[%d]: %d\n", i, bufferIDs[i]);

		alGenSources(1, &source);
		AL_CHECK("alGenSource() failed");

		{// build a sine wave in "audioBuffer"
			int NSAMPLES = sizeof(audioBuffer) / sizeof(audioBuffer[0]);
			for (int i = 0; i < NSAMPLES; i++) {
				double value = 2.0 * (double)i / (double)128 * M_PI;
				audioBuffer[i] = (short)(sin(value) * 32767.0);
			}
		}

		alBufferData(bufferIDs[0], AL_FORMAT_STEREO16, audioBuffer, sizeof(audioBuffer), sampleRate);
		AL_CHECK("alBufferData() failed");
		alBufferData(bufferIDs[1], AL_FORMAT_STEREO16, audioBuffer, sizeof(audioBuffer), sampleRate);
		AL_CHECK("alBufferData() failed");
		alBufferData(bufferIDs[2], AL_FORMAT_STEREO16, audioBuffer, sizeof(audioBuffer), sampleRate);
		AL_CHECK("alBufferData() failed");
		alBufferData(bufferIDs[3], AL_FORMAT_STEREO16, audioBuffer, sizeof(audioBuffer), sampleRate);
		AL_CHECK("alBufferData() failed");

		alSourceQueueBuffers(source, XAV_NUM_FRAMES, bufferIDs);
		AL_CHECK("alSourceQueueBuffers() failed");

		alSourcei(source, AL_LOOPING, AL_FALSE);
		AL_CHECK("alSourcei() failed");

		alSourcePlay(source);
		AL_CHECK("alSourcePlay() failed");
	}

	void Run() {
		int totalBytes = 0;
		int bufferIdx = 0;
		ALuint bufferId;
		ALint val;

		while (IsRunning()) {
			int idx = bufferIdx & (XAV_NUM_FRAMES - 1);
			XAVBuffer audio = stream->GetBuffer();
			AudioSampleFloat *pasf = (AudioSampleFloat *)audio.buffer;
			AudioSampleShort *pass = (AudioSampleShort *)audioBuffers[idx];

			// convert to signed short
			for (int i = 0; i < AUDIO_SAMPLES; i++) {
				pass->left = (short)(pasf->left * 32767.0f);
				pass->right = (short)(pasf->left * 32767.0f);
				pass++;
				pasf++;
			}

			totalBytes += audio.count;
			bufferIdx++;

			alGetSourcei(source, AL_BUFFERS_PROCESSED, &val);
			if (val > 0){
				alSourceUnqueueBuffers(source, 1, &bufferId);
				error = alGetError();
				if (error == AL_NO_ERROR) {
					alBufferData(bufferId, AL_FORMAT_STEREO16, audioBuffers[idx], AUDIO_SAMPLES * 4, sampleRate);
					alSourceQueueBuffers(source, 1, &bufferId);
				}
				val--;
			}
		}
	}

	std::shared_ptr<XAVStream> stream;

	ALCdevice *audioDevice;
	ALCcontext *audioContext;
	ALuint bufferIDs[XAV_NUM_FRAMES];
	ALenum error;
	ALuint source = 0;
	int sampleRate;
	short audioBuffer[AUDIO_SAMPLES];
	AudioSampleShort audioBuffers[XAV_NUM_FRAMES][AUDIO_SAMPLES];
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
	std::string videoPath = pathToAssets + "/assets/CulturalPhenomenon.mp4";

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(imgPath); return shape; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(9.6f, 5.4f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 0, 5.4f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		if (pavp != NULL && pavp->IsRunning()) {
			unsigned char *image = ib.b;

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 960, 540, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)image);
			GL_CHECK("glGetTexImage() didn't work");
		}
	};
	shape->SetTheFunk(transform);

	pavp = new AVPlayer(videoPath);
	pavp->Start();
}
