/**************************************************************
** MP4DevBuildScene.cpp
**
** Minimal libavcodec/libavformat demo, capable of decoding
** & rendering GoPro files shot in HD @ 120 FPS
**
** At present: 1920x1080 120 fps specifically.  All else: YMMV
**************************************************************/
#include "ExampleXGL.h"
#include "xav.h"

static const int numFrames = 128;
static const int vWidth = 1920;
static const int vHeight = 1080;

class VideoFrame {
public:
	VideoFrame(int ySize, int uvSize) {
		y = new uint8_t[ySize];
		u = new uint8_t[uvSize];
		v = new uint8_t[uvSize];
	}

	uint8_t *y, *u, *v;
};

class FramePool {
public:
	FramePool(int width, int height) : freeBuffs(numFrames), usedBuffs(0) {
		int ySize = width * height;
		int uvSize = (width / 2 * height / 2);

		for (int i = 0; i < numFrames; i++)
			frames[i] = new VideoFrame(ySize, uvSize);
	}

	~FramePool() {
		for (int i = 0; i < numFrames; i++)
			delete frames[i];
	}

	VideoFrame *frames[numFrames];
	XSemaphore freeBuffs, usedBuffs;
};

class MP4Demux  : public XThread {
public:
	MP4Demux(const char *fn) : fileName(fn), XThread("MP4Demux") {
		av_register_all();

		ended = true;
		StartPlaying();
	}

	~MP4Demux() {
		if (playing)
			StopPlaying();

		if (pCodecCtx)
			ReleaseAllTheThings();
	}

	void GetAllTheThings() {
		if ((pFormatCtx = avformat_alloc_context()) == nullptr)
			throwXAVException("Unable to allocate AVFormatContext for:  " + fileName + "");

		if ((pFrame = av_frame_alloc()) == nullptr)
			throwXAVException("Unable to allocate AVFrame for: " + fileName + "");

		if (avformat_open_input(&pFormatCtx, fileName.c_str(), 0, NULL) != 0)
			throwXAVException("avformat_open_input failed: " + fileName + "\n");

		if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
			throwXAVException("avformat_find_stream_info failed: Couldn't find stream information " + fileName + "\n");

		av_new_packet(&packet, 0x100000);

		for (int i = 0; pFormatCtx->nb_streams; i++) {
			if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
				vStreamIdx = i;
				vStream = pFormatCtx->streams[i];
				float frameRate = (float)vStream->r_frame_rate.num / (float)vStream->r_frame_rate.den;
				float duration = (float)vStream->duration / frameRate;
				xprintf("frameRate: %0.4f, %0.4f\n", frameRate, duration);

				break;
			}
		}

		if (vStreamIdx == -1)
			throwXAVException("No video stream found in " + fileName);

		pCodecCtx = pFormatCtx->streams[vStreamIdx]->codec;

		if ((pCodec = avcodec_find_decoder(pCodecCtx->codec_id)) == nullptr)
			throwXAVException("Unsupported codec " + fileName + "!\n");

		if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
			throwXAVException("Unable to open codec " + fileName + "");

		timeBase = pCodecCtx->time_base;
		ticksPerFrame = pCodecCtx->ticks_per_frame;

		xprintf("video context delay: %d, num: %d, den: %d\n", pCodecCtx->delay, timeBase.num, timeBase.den);

		// figure out the chroma sub-sampling of the pixel format
		int pixelFormat = pCodecCtx->pix_fmt;
		const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get((AVPixelFormat)pixelFormat);

		// save it so our clients can know
		chromaWidth = pCodecCtx->width / (1 << pixDesc->log2_chroma_w);
		chromaHeight = pCodecCtx->height / (1 << pixDesc->log2_chroma_h);

		if ((pFrames = new FramePool(pCodecCtx->width, pCodecCtx->height)) == nullptr) {
			avcodec_close(pCodecCtx);
			avformat_close_input(&pFormatCtx);
			throwXAVException("Unable to allocate new FramePool " + fileName + "");
		}
	}

	void ReleaseAllTheThings(){
		avcodec_close(pCodecCtx);
		avformat_close_input(&pFormatCtx);

		if (pFrames){
			delete pFrames;
			pFrames = nullptr;
		}
		if (pFormatCtx) {
			avformat_free_context(pFormatCtx);
			pFormatCtx = nullptr;
		}
		if (pFrame) {
			av_frame_free(&pFrame);
			pFrame = nullptr;
		}

		av_shrink_packet(&packet, 0);
	}

	void Run() {
		while (IsRunning()) {
			if (playing || step) {
				std::unique_lock<std::mutex> lock(playMutex);
				if ((retVal = av_read_frame(pFormatCtx, &packet)) == 0) {
					if (packet.stream_index == vStreamIdx) {
						int frameFinished;
						avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
						if (frameFinished) {
							currentPlayTime = Pts2Time(pFrame->pkt_dts);
							int64_t pts = Time2Pts(currentPlayTime);
							if (showFrameStatus) {
								xprintf("pts: %d, %0.4f, %d, pict_type: %d%s\n", pFrame->pkt_pts, currentPlayTime, pts, pFrame->pict_type, pFrame->key_frame ? " key frame" : "");
								showFrameStatus = false;
							}
							if (pFrames->freeBuffs.wait_for(100)) {
								VideoFrame *pvf = pFrames->frames[nFramesDecoded & (numFrames - 1)];

								int ySize = pFrame->height * pFrame->linesize[0];
								int uvSize = chromaWidth * chromaHeight;

								memcpy(pvf->y, pFrame->data[0], ySize);
								memcpy(pvf->u, pFrame->data[1], uvSize);
								memcpy(pvf->v, pFrame->data[2], uvSize);

								nFramesDecoded++;
								pFrames->usedBuffs.notify();
							}
							av_frame_unref(pFrame);
						}
						if (step)
							step = false;
					}
					av_free_packet(&packet);
				}
				else {
					StopPlaying();
					ended = true;
					ReleaseAllTheThings();
				}
			}
			else
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
		}
	}

private:
	void Seek(int64_t timeOffset) {
		std::unique_lock<std::mutex> lock(playMutex);
		wasPlaying = playing;
		playing = false;

		pFrames->freeBuffs(0);
		pFrames->usedBuffs(0);
		nFramesDecoded = 0;
		nFramesDisplayed = 0;
		retVal = avformat_seek_file(pFormatCtx, -1, INT64_MIN, timeOffset, INT64_MAX, 0);
		avcodec_flush_buffers(pCodecCtx);
		showFrameStatus = true;
		pFrames->freeBuffs(numFrames);
		if (wasPlaying)
			playing = true;
	}

public:
	void SeekPercent(float percent) {
		if (!ended) {
			float duration = (float)pFormatCtx->duration;
			int64_t timeOffset = (int64_t)(duration * percent / 100.0f);
			Seek(timeOffset);
		}
	}

	void StartPlaying() {
		if (playing)
			return;

		if (ended) {
			GetAllTheThings();
			ended = false;
			nFramesDecoded = nFramesDisplayed = 0;
			pFrames->usedBuffs(0);
			pFrames->freeBuffs(numFrames);
		}
		playing = true; 
	}

	void StopPlaying() { 
		playing = false; 
	}

	float Pts2Time(int64_t pts){
		return ((float)pts * (float)(timeBase.num * ticksPerFrame) / (float)timeBase.den) / 1000.0f;
	}

	uint64_t Time2Pts(float time) {
		return (uint64_t)((time * 1000.0f) * (float)timeBase.den) / timeBase.num / ticksPerFrame;
	}

	std::string fileName;
	AVFormatContext *pFormatCtx = nullptr;
	AVStream *vStream = nullptr;
	int vStreamIdx{ -1 };
	AVCodecContext *pCodecCtx = nullptr;
	AVCodec *pCodec = nullptr;
	AVPacket packet{};
	AVFrame *pFrame = nullptr;
	AVRational timeBase;
	int ticksPerFrame;
	int retVal{ 0 };

	FramePool *pFrames = nullptr;
	int	nFramesDecoded{ 0 }, nFramesDisplayed{ 0 };
	int	chromaWidth{ 0 }, chromaHeight{ 0 };
	bool playing{ false };
	bool ended{ false };
	bool step{ false };
	bool wasPlaying{ false };
	bool showFrameStatus{ false };
	float currentPlayTime{ 0.0 };

	std::mutex playMutex;
};

MP4Demux *pMp4;
XGLShape *shape;

extern bool initHmd;

void ExampleXGL::BuildScene() {
	preferredSwapInterval = 1;
	preferredWidth = 1880;
	preferredHeight = 960;
	//initHmd = true;
	
	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string videoPath;
	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(vWidth, vHeight, 1); return shape; });
	shape->AddTexture(vWidth / 2, vHeight / 2, 1);
	shape->AddTexture(vWidth / 2, vHeight / 2, 1);

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate *scale;

	pMp4 = new MP4Demux(videoPath.c_str());

	shape->SetAnimationFunction([&](float clock) {
		static float oldClock = 0.0f;
		if (clock > oldClock) {
			oldClock = clock;
			if (pMp4->playing){
				if (pMp4->pFrames->usedBuffs.get_count() > 2) {
					VideoFrame *pFrame = pMp4->pFrames->frames[pMp4->nFramesDisplayed++ & (numFrames - 1)];

					glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit0"), 0);
					glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit1"), 1);
					glProgramUniform1i(shape->shader->programId, glGetUniformLocation(shape->shader->programId, "texUnit2"), 2);

					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, shape->texIds[0]);
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vWidth, vHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->y);
					GL_CHECK("glGetTexImage() didn't work");

					glActiveTexture(GL_TEXTURE1);
					glBindTexture(GL_TEXTURE_2D, shape->texIds[1]);
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pMp4->chromaWidth, pMp4->chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->u);
					GL_CHECK("glGetTexImage() didn't work");

					glActiveTexture(GL_TEXTURE2);
					glBindTexture(GL_TEXTURE_2D, shape->texIds[2]);
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pMp4->chromaWidth, pMp4->chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->v);
					GL_CHECK("glGetTexImage() didn't work");

					pMp4->pFrames->freeBuffs.notify();
				}
			}
		}
	});

	XInputKeyFunc seekFunc = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			pMp4->SeekPercent(0.0f);
		}
	};


	XInputKeyFunc seekPercentFunc = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			key -= '0';
			if (key < 0)
				key = 0;
			else if (key > 9)
				key = 9;

			pMp4->SeekPercent((float)key * 10.0f);
		}
	};
	AddKeyFunc(XInputKeyRange('0', '9'), seekPercentFunc);

	XInputKeyFunc seekDeltaFunc = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		int64_t time = -1;
		if (isDown && !isRepeat) {
			switch (key) {
				case GLFW_KEY_RIGHT:
					xprintf("+1\n");
					pMp4->playing = false;
					pMp4->step = true;
					break;

				case GLFW_KEY_LEFT:
					xprintf("-1\n");
					pMp4->playing = false;
					break;
			}
		}
	};
	AddKeyFunc(XInputKeyRange(GLFW_KEY_RIGHT, GLFW_KEY_END), seekDeltaFunc);

	AddKeyFunc('P', [&](int key, int flags){
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			pMp4->StartPlaying();
		}
	});

	AddKeyFunc('O', [&](int key, int flags){
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			pMp4->StopPlaying();
		}
	});

	XGLGuiManager *gm = GetGuiManager();
	XGLGuiSlider *slider = (XGLGuiSlider *)(gm->FindObject("File Seek"));
	if (slider != nullptr) {
		XGLGuiCanvas::MouseEventListener mel = [slider](float x, float y, int flags) {
			if (slider->HasMouse()) {
				XGLGuiCanvas *thumb = (XGLGuiCanvas *)slider->Children()[1];
				float percent = slider->Position() * 100.0f;
				pMp4->SeekPercent(percent);
			}
		};

		slider->AddMouseEventListener(mel);
	}

	pMp4->Start();
}
