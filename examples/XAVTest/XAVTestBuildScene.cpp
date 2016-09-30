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
	int width, height;
} ImageBuff;

ImageBuff ib;

class VideoStreamThread : public XThread {
public:
	VideoStreamThread(std::shared_ptr<XAVStream> s) : XThread("VideoStreamThread"), stream(s) {}

	void Run() {
		while (IsRunning()) {
			XAVBuffer image = stream->GetBuffer();
			if (image.buffer == NULL) {
				Stop();
				break;
			}
			memcpy(&ib.b, image.buffer, sizeof(ib.b));
			ib.width = stream->width;
			ib.height = stream->height;
		}
	}

	std::shared_ptr<XAVStream> stream;
};

class AudioStreamThread : public XThread {
public:
	AudioStreamThread(std::shared_ptr<XAVStream> s) : XThread("AudioStreamThread"), stream(s) {
		int sampleSize = stream->formatSize * stream->channels;

		sampleRate = stream->sampleRate;

		audioDevice = alcOpenDevice(NULL);
		AL_CHECK("alOpenDevice() failed");

		audioContext = alcCreateContext(audioDevice, NULL);
		AL_CHECK("alcCreateContext() failed\n");

		alcMakeContextCurrent(audioContext);
		alGetError();

		alGenBuffers(XAV_NUM_FRAMES, bufferIDs);
		AL_CHECK("alGenBuffers() failed");

		alGenSources(1, &source);
		AL_CHECK("alGenSource() failed");

		alSourcei(source, AL_LOOPING, AL_FALSE);
		AL_CHECK("alSourcei() failed");

		memset(audioBuffer, 0, sizeof(audioBuffer));

		for (int i = 0; i < XAV_NUM_FRAMES; i++) {
			alBufferData(bufferIDs[i], AL_FORMAT_STEREO16, audioBuffer, 4, sampleRate);
			AL_CHECK("alBufferData() failed");
		}

		alSourceQueueBuffers(source, XAV_NUM_FRAMES, bufferIDs);
		AL_CHECK("alSourceQueueBuffers() failed");
	}

	void Run() {
		int totalBytes = 0;
		int bufferIdx = 0;
		ALuint bufferId;
		ALint val;

		alSourcePlay(source);
		AL_CHECK("alSourcePlay() failed");

		while (IsRunning()) {
			do {
				alGetSourcei(source, AL_BUFFERS_PROCESSED, &val);
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
			} while (val == 0);

			alSourceUnqueueBuffers(source, 1, &bufferId);

			// need a better mapping for this, good enough development tests for now
			int idx = bufferId - 1;
			XAVBuffer audio = stream->GetBuffer();

			if (audio.buffer == NULL) {
				alSourceStop(source);
				Stop();
				break;
			}
			float *pLeft = (float *)audio.buffers[0];
			float *pRight = (float *)audio.buffers[1];
			AudioSampleShort *pass = (AudioSampleShort *)audioBuffers[idx];

			// convert to signed short
			for (int i = 0; i < AUDIO_SAMPLES; i++) {
				pass->left = (short)(*pLeft * 31000.0f);
				pass->right = (short)(*pRight * 31000.0f);
				pass++;
				pLeft++;
				pRight++;
			}

			totalBytes += audio.count;
			bufferIdx++;

			alBufferData(bufferId, AL_FORMAT_STEREO16, audioBuffers[idx], AUDIO_SAMPLES * 4, sampleRate);
			AL_CHECK("alBufferData() failed");
			alSourceQueueBuffers(source, 1, &bufferId);
			AL_CHECK("alSourceQueueBuffers() failed");

			// may need to restart here if initial queueing was too short 
			// ie: we ran out and stopped before we got here
			if (1) {
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
		xavSrc = std::make_shared<XAVSrc>(url, true, true);
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

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.width, ib.height, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)image);
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
