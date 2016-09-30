/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>
#include <xal.h>

#include <iostream>

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
			long long start = hpt.Count();
			long long end = start;
			long long diff = end - start;
	
			XAVBuffer image = stream->GetBuffer();
			if (image.buffer == NULL) {
				Stop();
				break;
			}
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
		int sampleSize = stream->formatSize * stream->channels;

		sampleRate = stream->sampleRate;

		if ((audioDevice = alcOpenDevice(NULL)) == NULL) {
			throwXGLException("alcOpenDevice() failed");
		}

		if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL) {
			throwXGLException("alcCreateContext() failed\n");
		}

		alcMakeContextCurrent(audioContext);
		alGetError();

		alGenBuffers(XAV_NUM_FRAMES, bufferIDs);
		AL_CHECK("alGenBuffers() failed");

		alGenSources(1, &source);
		AL_CHECK("alGenSource() failed");

		{// build a sine wave in "audioBuffer"
			int NSAMPLES = sizeof(audioBuffer) / sizeof(audioBuffer[0]);
			for (int i = 0; i < NSAMPLES; i++) {
				double value = 2.0 * (double)i / (double)128 * M_PI;
				audioBuffer[i] = (short)(sin(value) * 32767.0);
			}
		}

		for (int i = 0; i < XAV_NUM_FRAMES; i++) {
			alBufferData(bufferIDs[i], AL_FORMAT_STEREO16, audioBuffer, sizeof(audioBuffer), sampleRate);
			AL_CHECK("alBufferData() failed");
		}

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
			if (audio.buffer == NULL) {
				alSourceStop(source);
				Stop();
				break;
			}
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

			if (1) // dirty hack to work around the lack of robust buffering, (from OpenAL examples) DEFINITELY causes audio glitches.
			{
				ALint state;
				alGetSourcei(source, AL_SOURCE_STATE, &state);
				if (state != AL_PLAYING)
					alSourcePlay(source);
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
		vst = new VideoStreamThread(xavSrc->mVideoStream);
		ast = new AudioStreamThread(xavSrc->mAudioStream);
		xav.Start();
	}

	void Run() {
		vst->Start();
		ast->Start();

		while ( xav.IsRunning() && IsRunning() ) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
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
	XGLShape *shape, *childRed, *childYellow;

	std::string imgPath = pathToAssets + "/assets/AndroidDemo.png";
	std::string videoPath = pathToAssets + "/assets/CulturalPhenomenon.mp4";

	AddShape("shaders/lighting", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->SetColor({ 0.8, 0.8, 0.0001 });
	shape->SetTheFunk([&](XGLShape *s, float clock) {
		float translateFunction = sin(clock / 60.0f);
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(translateFunction*4.0f, 0.0f, 0.0f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 40.0f, glm::vec3(1.0f, 0.0f, 0.0f));

		s->model = translate * rotate;
	});

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(imgPath); return shape; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 4.0f, 5.625f));
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

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(5, -20, 20);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	pavp = new AVPlayer(videoPath);
	pavp->Start();
}
