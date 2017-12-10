/**************************************************************
** XAVTestBuildScene.cpp
**
** XAV testing scene.
**
** NOTE: The preferred way to fetch from data streams (not audio/video)
** is to have GUI functions pull from data resources that are
** managed by the DataStreamThread objects, rather than have
** the DataStreamThread objects try and "push" their data to
** a GUI element.  Threads are not shut down cleanly at the
** moment, and are still running when the XGL object is being
** destroyed.  So the data threads can't be referencing GUI
** stuff, because it won't be there during program shutdown.
**************************************************************/
#include "ExampleXGL.h"

#include <iostream>

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>
#include <xal.h>
#include "xavdata.h"
#include "xavgpmf.h"

extern bool initHmd;

#define VIDEO_WIDTH 2704
#define VIDEO_HEIGHT 2624
#define VIDEO_CHANNELS 3

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
	VideoStreamThread(XGL* pXgl, std::shared_ptr<XAVStream> s) : pXgl(pXgl), XThread("VideoStreamThread"), stream(s) {
		memset(ib.y, 0, sizeof(ib.y));
		memset(ib.u, 127, sizeof(ib.u));
		memset(ib.v, 127, sizeof(ib.v));
		ib.width = VIDEO_WIDTH;
		ib.height = VIDEO_HEIGHT;
		ib.chromaWidth = VIDEO_WIDTH / 2;
		ib.chromaHeight = VIDEO_HEIGHT / 2;
		penX = 0;
		penY = 34;
	}

	void Run() {
		try {
			int size;

			while (IsRunning()) {
				XAVStream::XAVBuffer image = stream->GetBuffer();
				if (image.buffers[0] == NULL) {
					Stop();
					break;
				}

				size = stream->width * stream->height;
				if (size > (VIDEO_WIDTH*VIDEO_HEIGHT))
					throwXAVException(" Buffer size exceeded. Ensure that\n VIDEO_WIDTH & VIDEO_HEIGHT\n are sufficient for the video chosen.");

				memcpy(&ib.y, image.buffers[0], size);

				size = stream->chromaWidth * stream->chromaHeight;
				memcpy(&ib.u, image.buffers[1], size);
				memcpy(&ib.v, image.buffers[2], size);

				ib.width = stream->width;
				ib.height = stream->height;
				ib.chromaWidth = stream->chromaWidth;
				ib.chromaHeight = stream->chromaHeight;
			}
			xprintf("VideoStreamThread done.\n");
		}
		catch (std::runtime_error e) {
			// render the exception what() string into the video buffer.
			font.SetPixelSize(32);
			font.RenderText(std::string("VideoStreamThread:\n") + e.what(), ib.y, ib.width, ib.height, &penX, &penY);

			Stop();
		}
	}

	std::shared_ptr<XAVStream> stream;
	int penX;
	int penY;
	XGL *pXgl;
};

class AudioStreamThread : public XThread {
public:
	AudioStreamThread(XGL* pXgl, std::shared_ptr<XAVStream> s) : pXgl(pXgl), XThread("AudioStreamThread"), stream(s), xal(NULL, s->sampleRate, XAL::defaultFormat, XAVStream::numFrames) {
		xal.AddBuffers(XAVStream::numFrames);
		xal.QueueBuffers();
		xal.Play();
	}

	void Run() {
		while (IsRunning()) {
			xal.WaitForProcessedBuffer();

			XAVStream::XAVBuffer audio = stream->GetBuffer();

			if (audio.buffers[0] == NULL) {
				xal.Stop();
				Stop();
				break;
			}

			xal.Convert((float *)audio.buffers[0], (float *)audio.buffers[1]);
			xal.Buffer();
			xal.Restart();
		}
		xprintf("AudioStreamThread done.\n");
	}

	XAL xal;
	std::shared_ptr<XAVStream> stream;
	XGL *pXgl;
};

class AVPlayer : public XObject, public XThread {
public:
	AVPlayer(XGL *pXgl, std::string url) : pXgl(pXgl), XObject("AVPlayer"), XThread("AVPlayerThread"), textWindow(nullptr) {
		// Video can twiddle the HUD/GUI "TextWindow" object
		// (remember to consider thread safety!)
		if ((textWindow = (XGLGuiWindow*)pXgl->FindObject("GuiTextWindow"))) {
			xprintf("Found 'GuiTextWindow'\n");
		}

		// once XAVSrc is constructed, it has parsed the stream looking for video & audio
		// (or else it threw an exception)
		xavSrc = std::make_shared<XAVSrc>(url, true, true);

		hasVideo = xavSrc->mVideoStream != NULL;
		hasAudio = xavSrc->mAudioStream != NULL;

		xav.AddSrc(xavSrc);

		if (hasVideo)
			vst = new VideoStreamThread(pXgl, xavSrc->mVideoStream);

		if (hasAudio)
			ast = new AudioStreamThread(pXgl, xavSrc->mAudioStream);
		
		// add additional streams as generic data streams
		for (size_t i = 2; i < xavSrc->mStreams.size(); i++)
			dataStreamThreads.emplace_back(new XAVDataThread(xavSrc->mStreams[i]));

		xav.Start();
	}

	~AVPlayer() {
		xprintf("AVPlayer::~AVPlayer() \n");
		Stop();
	}

	void Run() {
		if (hasAudio)
			ast->Start();

		if (hasVideo)
			vst->Start();

		for (auto dst : dataStreamThreads)
			dst->Start();

		while ( xav.IsRunning() && IsRunning() ) {
			if (hasVideo && !vst->IsRunning())
				break;
			if (hasAudio && !ast->IsRunning())
				break;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
		}

		for (auto dst : dataStreamThreads)
			dst->Stop();

		if (hasVideo)
			vst->Stop();

		if (hasAudio)
			ast->Stop();
	}

	XGL* pXgl;
	XAV xav;
	VideoStreamThread *vst;
	AudioStreamThread *ast;

	std::shared_ptr<XAVSrc> xavSrc;
	bool hasVideo, hasAudio;
	
	XAVDataThreads dataStreamThreads;
	XGLGuiWindow *textWindow;
};

namespace {
	AVPlayer *pavp;
};

XGLGuiWindow* textWindow = nullptr;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	//initHmd = true;

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(5, -20, 20);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	// build a full path including "pathToAssets", unless it's a url that starts with "http"
	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string videoPath;
	if (videoUrl.find("http") != std::string::npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	if (false){
		AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
		shape->attributes.diffuseColor = { 0.025, 0.025, 0.025, 1 };
		shape->SetAnimationFunction([shape](float clock) {
			float translateFunction = sin(clock / 60.0f);
			glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(translateFunction*4.0f, 0.0f, 0.0f));
			glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 40.0f, glm::vec3(1.0f, 0.0f, 0.0f));
			shape->model = translate * rotate;
		});
	}

	AddShape("shaders/yuv", [&](){ shape = new XGLHemiSphere(1.0f, 256); return shape; });
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(180.f), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(20.0f, 20.0f, 20.0f));
	shape->model = translate * scale * rotate;

	bool doVideo = true;
	if (doVideo) {
		AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(VIDEO_WIDTH, VIDEO_HEIGHT, 1); return shape; });
		shape->AddTexture(VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2, 1);
		shape->AddTexture(VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2, 1);

		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.00001f, 0.00001f, 0.00001f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 2.624f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate * scale;

		shape->SetAnimationFunction([shape](float clock) {
			if (pavp != NULL && pavp->IsRunning() && (ib.width != 0)) {
				glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit0"), 0);
				glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit1"), 1);
				glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit2"), 2);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, shape->texIds[0]);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.width, ib.height, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.y);
				GL_CHECK("glGetTexImage() didn't work");

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, shape->texIds[1]);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.chromaWidth, ib.chromaHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.u);
				GL_CHECK("glGetTexImage() didn't work");

				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, shape->texIds[2]);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, ib.chromaWidth, ib.chromaHeight, 0, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)ib.v);
				GL_CHECK("glGetTexImage() didn't work");
			}
		});

		pavp = new AVPlayer(this, videoPath);
		pavp->SetName("AVPlayer");

		// setup data callback
		// Assume GoPro stream for now.
		if (pavp->dataStreamThreads.size() >= 2) {
			// Get the GPMF user data stream
			XAVDataThread *pdt = pavp->dataStreamThreads[1];

			XAVDataListener gpmfParser = [&](void *ctx, XCircularBuffer *pcb) {
				XAVDataThread *pdt = (XAVDataThread *)ctx;
				int count = pcb->Count() >> 2; // this is what GPMF_Init() uses
				uint32_t *buff = new uint32_t[count];
				uint32_t *pBuff = buff;
				pcb->Read((uint8_t*)buff, count<<2);

				pdt->UpdateStatus("");
				pdt->UpdateStatus(" ");

				// skip to first DEVC key, (probably a no-op, but just to be safe)
				for (; *pBuff != MAKEID('D', 'E', 'V', 'C') && pBuff < (buff + count); pBuff++);

				for (; pBuff < (buff + count); pBuff++) {
					uint32_t key = *pBuff;
					uint8_t *p8Buff = (uint8_t*)(pBuff + 1);
					GPMF_TypeSizeLength kvl;

					kvl.type = *(p8Buff++);
					kvl.size = *(p8Buff++);
					kvl.count = *(p8Buff++) << 8;
					kvl.count += *(p8Buff++);
					pBuff++;

					xprintf("%c%c%c%c - '%c', %d, %d\n", PRINTF_4CC(key), kvl.type?kvl.type:'0', kvl.size, kvl.count);
					pBuff += (kvl.count + 3) >> 2;
				}

				delete buff;
			};

			pdt->AddListener(gpmfParser);

			AddShape("shaders/000-simple", [&](){ shape = new XGLTransformer(); return shape; });

			XGLShape::AnimationFn fn = [this](float clock) {
				if (pavp->textWindow) {
					XAVDataThread *pdt = pavp->dataStreamThreads[1];
					std::string s = pdt->Status();
					if (s.size()) {
						pavp->textWindow->Clear();
						pavp->textWindow->SetPenPosition(10, 16);
						pavp->textWindow->RenderText(s, 16);
						pdt->UpdateStatus("");
					}
				}
			};

			shape->SetAnimationFunction(fn);
		}
		pavp->Start();
	}
}
