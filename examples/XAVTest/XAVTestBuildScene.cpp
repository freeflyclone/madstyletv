/**************************************************************
** XAVTestBuildScene.cpp
**
**************************************************************/
#include "ExampleXGL.h"

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>
#include <xal.h>
#include <xtimer.h>

#include <iostream>

#define VIDEO_WIDTH 1920
#define VIDEO_HEIGHT 1080
#define VIDEO_CHANNELS 4

typedef struct {
	unsigned char y[VIDEO_WIDTH * VIDEO_HEIGHT];
	unsigned char u[VIDEO_WIDTH * VIDEO_HEIGHT];
	unsigned char v[VIDEO_WIDTH * VIDEO_HEIGHT];
	int width, height;
	int chromaWidth, chromaHeight;
} ImageBuff;

ImageBuff ib;

class VideoStreamThread : public XThread {
public:
	VideoStreamThread(std::shared_ptr<XAVStream> s) : XThread("VideoStreamThread"), stream(s) {}

	void Run() {
		while (IsRunning()) {
			XAVBuffer image = stream->GetBuffer();
			if (image.buffers[0] == NULL) {
				Stop();
				break;
			}
			memcpy(&ib.y, image.buffers[0], sizeof(ib.y));
			memcpy(&ib.u, image.buffers[1], sizeof(ib.u));
			memcpy(&ib.v, image.buffers[2], sizeof(ib.v));
			ib.width = stream->width;
			ib.height = stream->height;
			ib.chromaWidth = stream->chromaWidth;
			ib.chromaHeight = stream->chromaHeight;
			std::this_thread::sleep_for(std::chrono::duration<int, std::micro>(1));
			double sinceLast = 0.0;
			do {
				sinceLast += timer.SinceLast();
			} while (sinceLast < 0.016666);
		}
	}

	std::shared_ptr<XAVStream> stream;
	XTimer timer;
};

class AudioStreamThread : public XThread {
public:
	AudioStreamThread(std::shared_ptr<XAVStream> s) : XThread("AudioStreamThread"), stream(s), pXal(NULL) {
		pXal = new XAL(NULL, stream->sampleRate, AL_FORMAT_STEREO16, XAV_NUM_FRAMES);
		pXal->AddBuffers(XAV_NUM_FRAMES);
		pXal->QueueBuffers(4);
		pXal->Play();
	}

	~AudioStreamThread() {
		delete pXal;
	}

	void Run() {
		while (IsRunning()) {
			pXal->WaitForProcessedBuffer();

			XAVBuffer audio = stream->GetBuffer();

			if (audio.buffers[0] == NULL) {
				pXal->Stop();
				Stop();
				break;
			}

			pXal->Convert((float *)audio.buffers[0], (float *)audio.buffers[1]);
			pXal->Buffer();
			pXal->Restart();
		}
	}

	std::shared_ptr<XAVStream> stream;
	XAL *pXal;
};

class AVPlayer : public XGLObject, public XThread {
public:
	AVPlayer(std::string url) : XGLObject("AVPlayer"), XThread("AVPlayerThread") {
		xavSrc = std::make_shared<XAVSrc>(url, true, true);
		// once xavSrc has been constructed, it has parsed the stream looking for video & audio
		hasVideo = xavSrc->mVideoStream != NULL;
		hasAudio = xavSrc->mAudioStream != NULL;

		xav.AddSrc(xavSrc);

		if (hasVideo)
			vst = new VideoStreamThread(xavSrc->mVideoStream);

		if (hasAudio)
			ast = new AudioStreamThread(xavSrc->mAudioStream);

		xav.Start();
	}

	void Run() {
		if (hasVideo)
			vst->Start();

		if (hasAudio)
			ast->Start();

		while ( xav.IsRunning() && IsRunning() ) {
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
		}

		if (hasVideo)
			vst->Stop();

		if (hasAudio)
			ast->Stop();
	}

	XAV xav;
	VideoStreamThread *vst;
	AudioStreamThread *ast;
	std::shared_ptr<XAVSrc> xavSrc;
	bool hasVideo, hasAudio;
};

AVPlayer *pavp;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	std::string videoPath = pathToAssets + "/" + config.WideToBytes(config.Find(L"VideoFile")->AsString());

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = { 0.025, 0.025, 0.025, 1 };
	shape->SetTheFunk([&](XGLShape *s, float clock) {
		float translateFunction = sin(clock / 60.0f);
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(translateFunction*4.0f, 0.0f, 0.0f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 40.0f, glm::vec3(1.0f, 0.0f, 0.0f));

		s->model = translate * rotate;
	});

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(VIDEO_WIDTH,VIDEO_HEIGHT,1); return shape; });
	shape->AddTexture(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, 1);
	shape->AddTexture(VIDEO_WIDTH/2, VIDEO_HEIGHT/2, 1);

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(10.0f, 5.625f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0, 4.0f, 5.625f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate * scale;

	XGLShape::AnimaFunk transform = [&](XGLShape *s, float clock) {
		if (pavp != NULL && pavp->IsRunning() && (ib.width != 0)) {

			glProgramUniform1i(s->shader->programId, glGetUniformLocation(s->shader->programId, "texUnit0"), 0);
			glProgramUniform1i(s->shader->programId, glGetUniformLocation(s->shader->programId, "texUnit1"), 1);
			glProgramUniform1i(s->shader->programId, glGetUniformLocation(s->shader->programId, "texUnit2"), 2);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, s->texIds[0]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.width, ib.height, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.y);
			GL_CHECK("glGetTexImage() didn't work");

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, s->texIds[1]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.chromaWidth, ib.chromaHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.u);
			GL_CHECK("glGetTexImage() didn't work");

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, s->texIds[2]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.chromaWidth, ib.chromaHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.v);
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
