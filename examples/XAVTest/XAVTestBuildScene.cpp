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
#include <chrono>

#include <xav.h>
#include <xavfile.h>
#include <xfifo.h>
#include <xal.h>
#include <xtimer.h>
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
	float pts;
} ImageBuff;

ImageBuff ib;
typedef std::deque<ImageBuff> ImageBuffs;

class VideoStreamThread : public XThread {
public:
	VideoStreamThread(XGL* pXgl, std::shared_ptr<XAVStream> s) : pXgl(pXgl), XThread("VideoStreamThread"), stream(s), freeBuffs(2), usedBuffs(1) {
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

				framePeriod = (float)stream->framerateNum / (float)stream->framerateDen;
				pts = ((float)image.pts / framePeriod) / (float)stream->framerateDen;
				//xprintf("video pts: %0.3f\n", pts);

				freeBuffs.wait();
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
				ib.pts = pts;
				usedBuffs.notify();
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
	float pts;
	float framePeriod;

	XSemaphore freeBuffs, usedBuffs;
};

class AudioStreamThread : public XThread {
public:
	AudioStreamThread(XGL* pXgl, std::shared_ptr<XAVStream> s) : pXgl(pXgl), XThread("AudioStreamThread"), stream(s), xal(NULL, s->sampleRate, XAL::defaultFormat, XAL::maxBuffers) {
		xal.AddBuffers(XAL::maxBuffers);
		xal.QueueBuffers();
		xal.Play();

		float delta = (float)XAL::audioSamples / stream->sampleRate;
		deltaPts = delta;
	}

	void Run() {
		float channelBuff[XAVStream::maxChannels][XAL::audioSamples];
		auto start = std::chrono::steady_clock::now();

		while (IsRunning()) {
			{
				auto start = std::chrono::high_resolution_clock::now();
				xal.WaitForProcessedBuffer();
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::micro> duration = end - start;
				//xprintf("xal delay: %0.2fus\n", duration);
			}

			for (int i = 0; i < stream->channels; i++)
				stream->cbSet[i].get()->Read((uint8_t *)&channelBuff[i], XAL::audioSamples * stream->formatSize);

			xal.Convert(channelBuff[0], channelBuff[1]);
			xal.Buffer();
			xal.Restart();
			pts += deltaPts;

			auto now = std::chrono::steady_clock::now();
			std::chrono::duration<double, std::micro> when = now - start;

			//xprintf("%0.6f,%0.2fus\n", pts,when);
		}
		xprintf("AudioStreamThread done.\n");
	}

	XAL xal;
	std::shared_ptr<XAVStream> stream;
	XGL *pXgl;
	float pts{ 0.0f };
	float deltaPts;
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
			dataStreamThreads.emplace_back(new XAVGpmfThread(xavSrc->mStreams[i]));

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

		if (dataStreamThreads.size() > 0) {
			for (auto dst : dataStreamThreads)
				dst->Start();
		}

		while ( xav.IsRunning() && IsRunning() ) {
			if (hasVideo && !vst->IsRunning())
				break;
			if (hasAudio && !ast->IsRunning())
				break;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
		}

		if (dataStreamThreads.size() > 0) {
			for (auto dst : dataStreamThreads)
				dst->Stop();
		}

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
	
	XAVGpmfThreads dataStreamThreads;
	XGLGuiWindow *textWindow;
};

namespace {
	AVPlayer *pavp;

#define DATA_STREAMS_ENGAGE
#ifdef DATA_STREAMS_ENGAGE
	XAVGpmfTelemetry telemetry;
#endif
};

XGLGuiWindow* textWindow = nullptr;
SteppedTimer videoTimer;
GameTime videoTime;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	initHmd = false;
	preferredSwapInterval = 0;

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(5, -20, 20);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	// build a full path including "pathToAssets", unless it's a url that starts with "http"
	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string videoPath;
	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
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

	//AddShape("shaders/yuv", [&](){ shape = new XGLHemiSphere(1.0f, 256); return shape; });
	//glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 1.0f, 0.0f));
	//glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(180.f), glm::vec3(0.0f, 0.0f, 1.0f));
	//glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(20.0f, 20.0f, 20.0f));
	//shape->model = translate * scale * rotate;

	bool doVideo = true;
	if (doVideo) {
		AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(VIDEO_WIDTH, VIDEO_HEIGHT, 1); return shape; });

		// these are for the u & v chroma channels 
		// the luma channel is the primary texture generated by 
		// XGLTexQuad constructor
		shape->AddTexture(VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2, 1);
		shape->AddTexture(VIDEO_WIDTH / 2, VIDEO_HEIGHT / 2, 1);

		//glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.00001f, 0.00001f, 0.00001f));
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
		glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 8.0f));
		glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = translate * rotate *scale;

		shape->SetAnimationFunction([&,shape](float clock) {
			static float oldClock = 0.0f;
			if (clock > oldClock) {
				VideoStreamThread* pVst = pavp->vst;
				AudioStreamThread* pAst = pavp->ast;

				while (videoTimer.TryAdvance(videoTime)) {
					if (pavp != NULL && pVst->IsRunning() && (ib.width != 0)) {
						pVst->usedBuffs.wait();
						//xprintf("%0.5f, %0.5f\n", pVst->pts, pAst->pts);

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
						pVst->freeBuffs.notify();
					}
				}
				oldClock = clock;
			}
		});

		pavp = new AVPlayer(this, videoPath);
		pavp->SetName("AVPlayer");

//#define DATA_STREAMS_ENGAGE
#ifdef DATA_STREAMS_ENGAGE
		// setup data callback
		// Assume GoPro stream for now.
		if (pavp->dataStreamThreads.size() >= 2) {
			telemetry.InitListeners(pavp->dataStreamThreads);

			// clear default generic listener
			pavp->dataStreamThreads[1]->AddGenericListener(nullptr);

			// add DEVC listener
			XAVGpmfListener fn = [this](uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff) { 
				telemetry.UpdateStatus("DEVC\n");
				telemetry.PrintGPMF(key, tsl); 
			};
			pavp->dataStreamThreads[1]->AddListener(MAKEID('D','E','V','C'), fn);

			fn = [this](uint32_t key, GPMF_TypeSizeLength tsl, uint8_t* buff) { 
				telemetry.UpdateStatus("Stream: ");
				telemetry.UpdateStatus((char *)buff);
				telemetry.UpdateStatus("\n");
				xprintf("Stream: %s\n", buff);
			};
			pavp->dataStreamThreads[1]->AddListener(MAKEID('S', 'T', 'N', 'M'), fn);

			// add invisible XGLTransformer to attach a top-level XGLShape that we can add a callback
			// to for querying the GPMF telemetry module per video frame time, ie: pull data rather
			// than have engine threads pushing data to GUI elements.
			AddShape("shaders/000-simple", [&](){ shape = new XGLTransformer(); return shape; });

			XGLShape::AnimationFn afn = [this](float clock) {
				if (pavp->textWindow) {
					XAVGpmfThread *pgmf = pavp->dataStreamThreads[1];
					std::string s = telemetry.status;
					if (s.size()) {
						pavp->textWindow->Clear();
						pavp->textWindow->SetPenPosition(10, 16);
						pavp->textWindow->RenderText(s, 16);
						telemetry.UpdateStatus("");
					}
				}
			};

			shape->SetAnimationFunction(afn);
		}
#endif // DATA_STREAMS_ENGAGE
	
		pavp->Start();
		videoTimer.SetStepFrequency(120);
	}
}
