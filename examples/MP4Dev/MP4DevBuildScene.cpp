/**************************************************************
** MP4DevBuildScene.cpp
**
** Minimal libavcodec/libavformat demo, capable of decoding
** & rendering GoPro files shot in HD @ 120 FPS
**
** At present: 1920x1080 120 fps specifically.  All else: YMMV
**
** NOTE!:
**   Uh... at present (4/26/19) this requires an old version of
** libavcodec, libavformat, etc in order to achieve satisfying
** decode frame rates.  I don't know what's changed in later versions,
** and it's quite possibly pilot error on my part, but I'm 
** looking at it.
**************************************************************/
#include "ExampleXGL.h"
#include "xav.h"

static const int numFrames = 4;
static const int vWidth = 1920;
static const int vHeight = 1080;

class VideoFrameBuffer {
public:
	VideoFrameBuffer(int ySize, int uvSize) : ySize(ySize), uvSize(uvSize) {
		y = new uint8_t[ySize];
		u = new uint8_t[uvSize];
		v = new uint8_t[uvSize];
	}

	uint8_t *y, *u, *v;
	int ySize, uvSize;
};

class FrameBufferPool {
public:
	FrameBufferPool(int width, int height) : freeBuffs(numFrames), usedBuffs(0) {
		int ySize = width * height;
		int uvSize = (width / 2 * height / 2);

		for (int i = 0; i < numFrames; i++)
			frames[i] = new VideoFrameBuffer(ySize, uvSize);
	}

	~FrameBufferPool() {
		for (int i = 0; i < numFrames; i++)
			delete frames[i];
	}

	VideoFrameBuffer* NextFree() { 
		if (!freeBuffs.wait_for(100))
			return nullptr;

		return frames[(freeIdx++)&(numFrames - 1)];
	};
	
	VideoFrameBuffer* NextUsed() {
		if (!usedBuffs.wait_for(100))
			return nullptr;

		return frames[(usedIdx++)&(numFrames - 1)];
	};

	void NotifyFree(){
		freeBuffs.notify();
	};

	void NotifyUsed() {
		usedBuffs.notify();
	};

	void Flush() {
		usedBuffs(0);
		freeBuffs(numFrames);
		freeIdx = 0;
		usedIdx = 0;
	}

	VideoFrameBuffer *frames[numFrames];
	XSemaphore freeBuffs, usedBuffs;
	uint64_t freeIdx{ 0 };
	uint64_t usedIdx{ 0 };
};

// XAVPacket derives from AVPacket soley to add the assignment operator
// (not copying side_data for now)
class XAVPacket : public AVPacket {
public:
	void operator = (const XAVPacket& p) {
		std::lock_guard<std::mutex> lock(mutex);

		size = p.size;
		dts = p.dts;
		stream_index = p.stream_index;
		flags = p.flags;
		duration = p.duration;
		pos = p.pos;
		convergence_duration = p.convergence_duration;
		memcpy(data, p.data, size);
	}

	std::mutex mutex;
};

class XAVDemux : public XThread {
public:
	class Sequencer : public XThread, public SteppedTimer {
	public:
		Sequencer(XAVDemux& dmx) : XThread("XAVDemux::SequencerThread"), dmx(dmx) {
			SetStepFrequency(120);
		};

		void Run() {
			while (IsRunning()) {
				GameTime gameTime;
				while (TryAdvance(gameTime))
					dmx.UpdateDisplay();
			}
		}

		XAVDemux& dmx;
	} sequencer;

	XAVDemux(std::string fn) : fileName(fn), XThread("XAVDemux"), sequencer(*this) {
		av_register_all();

		ended = true;
		StartPlaying();
	}

	~XAVDemux() {
		if (playing)
			StopPlaying();

		// thread stop, not "playing stop"
		WaitForStop();

		if (pCodecCtx)
			ReleaseAllTheThings();
	}

	// might need some PBO mapping wizardry here, eventually.
	static void our_buffer_default_free(void *opaque, uint8_t *data) {

	}

	// AVCodecContext.get_buffer2() override: force use of our supplied buffers
	// for decoder output - ie: decode straight into texture map memory (eventually)
	// For now, just getting it to use FrameBufferPool buffers is proof that 
	// this can work.  Eventually, will be PBO buffers with intent to remove
	// any CPU -> GPU copying of decoder results
	static int GetBuffer2(struct AVCodecContext *s, AVFrame *frame, int flags) {
		XAVDemux *self = (XAVDemux *)s->opaque;
		int ret = 0;
		int count = 0;

		VideoFrameBuffer* pvfb = self->pFrames->NextFree();
		if (pvfb) {
			frame->data[0] = pvfb->y;
			frame->data[1] = pvfb->u;
			frame->data[2] = pvfb->v;

			frame->linesize[0] = s->width;
			frame->linesize[1] = self->chromaWidth;
			frame->linesize[2] = self->chromaWidth;

			frame->buf[0] = av_buffer_create(frame->data[0], frame->linesize[0] * frame->height, our_buffer_default_free, self, 0);
			frame->buf[1] = av_buffer_create(frame->data[1], frame->linesize[1] * frame->height / 2, our_buffer_default_free, self, 0);
			frame->buf[2] = av_buffer_create(frame->data[2], frame->linesize[2] * frame->height / 2, our_buffer_default_free, self, 0);

			self->pFrames->NotifyUsed();
		}

		// must return 0!  can't remember where I saw it, but 0 means NOT ref counted. (I think)
		// (basically, we want to tell libavcodec to NOT fuck with management of these buffers.)
		return 0;
	}

	void GetAllTheThings() {
		if ((pFrame = av_frame_alloc()) == nullptr)
			throwXAVException("Unable to allocate AVFrame for: " + fileName + "");

		if (avformat_open_input(&pFormatCtx, fileName.c_str(), 0, NULL) != 0)
			throwXAVException("avformat_open_input failed: " + fileName + "\n");

		if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
			throwXAVException("avformat_find_stream_info failed: Couldn't find stream information " + fileName + "\n");

		av_init_packet(&packet);
		packet.data = packetBuff;
		packet.size = sizeof(packetBuff);

		av_init_packet(&vPkt);
		vPkt.data = vPacketBuff;
		vPkt.size = sizeof(vPacketBuff);

		// find video & audio streams, save their indexes
		for (unsigned int i = 0; i<pFormatCtx->nb_streams; i++) {
			if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
				vStreamIdx = i;
				vStream = pFormatCtx->streams[i];
				float frameRate = (float)vStream->r_frame_rate.num / (float)vStream->r_frame_rate.den;
				float duration = (float)vStream->duration / frameRate;
				xprintf("frameRate: %0.4f, %0.4f\n", frameRate, duration);
			}
			else if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
				aStreamIdx = i;
			}
		}

		if (vStreamIdx == -1)
			throwXAVException("No video stream found in " + fileName);

		// get AVCodecContext for video stream, override its get_buffer2() method with
		// ours, and save "this" in its "opaque" pointer, for use by our GetBuffer2();
		// (override must be done before codec is opened)
		pCodecCtx = pFormatCtx->streams[vStreamIdx]->codec;
		pCodecCtx->get_buffer2 = GetBuffer2;
		pCodecCtx->opaque = this;

		if ((pCodec = avcodec_find_decoder(pCodecCtx->codec_id)) == nullptr)
			throwXAVException("Unsupported codec " + fileName + "!\n");

		if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
			throwXAVException("Unable to open codec " + fileName + "");

		// save timing information (so we stand a chance of appropriate playback speed)
		timeBase = pCodecCtx->time_base;
		ticksPerFrame = pCodecCtx->ticks_per_frame;

		xprintf("video context delay: %d, num: %d, den: %d\n", pCodecCtx->delay, timeBase.num, timeBase.den);

		// figure out the chroma sub-sampling of the pixel format
		int pixelFormat = pCodecCtx->pix_fmt;
		const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get((AVPixelFormat)pixelFormat);

		// save it so our clients can know
		chromaWidth = pCodecCtx->width / (1 << pixDesc->log2_chroma_w);
		chromaHeight = pCodecCtx->height / (1 << pixDesc->log2_chroma_h);

		// get new FrameBufferPool(), and tell it the coded size needed
		if ((pFrames = new FrameBufferPool(pCodecCtx->coded_width, pCodecCtx->coded_height)) == nullptr) {
			avcodec_close(pCodecCtx);
			avformat_close_input(&pFormatCtx);
			throwXAVException("Unable to allocate new FrameBufferPool " + fileName + "");
		}
	}

	void ReleaseAllTheThings(){
		if (pCodecCtx) {
			avcodec_close(pCodecCtx);
			pCodecCtx = nullptr;
		}

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

	// this is run by the sequencer thread, at PTS/DTS interval.
	// This copies into vFrameBuffer at the above interval,
	// INDEPENDENT of display refresh rate.
	void UpdateDisplay() {
		VideoFrameBuffer* pvfb = pFrames->NextUsed();

		if (pvfb) {
			std::lock_guard<std::mutex> lock(displayMutex);

			memcpy(vFrameBuffer.y, pvfb->y, vFrameBuffer.ySize);
			memcpy(vFrameBuffer.u, pvfb->u, vFrameBuffer.uvSize);
			memcpy(vFrameBuffer.v, pvfb->v, vFrameBuffer.uvSize);

			pFrames->NotifyFree();
		}
	}

	void Run() {
		while (IsRunning()) {
			if (playing) {
				std::lock_guard<std::mutex> lock(playMutex);
				if ((retVal = av_read_frame(pFormatCtx, &packet)) == 0) {
					if (packet.stream_index == vStreamIdx) {
						// copy the XAVPacket
						vPkt = packet;
						int frameFinished;
						avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &vPkt);
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
		std::lock_guard<std::mutex> lock(playMutex);
		wasPlaying = playing;
		playing = false;

		pFrames->Flush();

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
			pFrames->Flush();
		}
		sequencer.Start();
		playing = true;
	}

	void StopPlaying() {
		sequencer.Stop();
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
	int aStreamIdx{ -1 };
	AVCodecContext *pCodecCtx = nullptr;
	AVCodec *pCodec = nullptr;
	XAVPacket packet{}, vPkt, aPkt, uPkt[3];
	AVFrame *pFrame = nullptr;
	AVRational timeBase;
	int ticksPerFrame;
	int retVal{ 0 };

	//TODO(?): make these dynamic
	// AVPacket::size is the amount of active data in the packet,
	// NOT the size of the buffer.  The sizes of these were
	// empirically determined to be "big enough", but maybe
	// not for 4K I-frames.
	uint8_t packetBuff[0x100000];
	uint8_t vPacketBuff[0x100000];

	FrameBufferPool *pFrames = nullptr;
	VideoFrameBuffer vFrameBuffer{ vWidth*vHeight, (vWidth / 2)*(vHeight / 2) };

	int	chromaWidth{ 0 }, chromaHeight{ 0 };
	bool playing{ false };
	bool ended{ false };
	bool wasPlaying{ false };
	bool showFrameStatus{ false };
	float currentPlayTime{ 0.0 };

	std::mutex playMutex;
	std::mutex displayMutex;
};

class XAVPlayer : public XGLTexQuad {
public:
	XAVPlayer(std::string url) : dmx(url), XGLTexQuad(vWidth, vHeight, 1) {
		AddTexture(vWidth / 2, vHeight / 2, 1);
		AddTexture(vWidth / 2, vHeight / 2, 1);
	}

	~XAVPlayer() {
		dmx.Stop();
	}

	void StartPlaying() {
		if (dmx.IsRunning() == false)
			dmx.Start();

		dmx.StartPlaying();
	}

	void StopPlaying() {
		dmx.StopPlaying();
	}

	void Draw() {
		if (dmx.pFrames) {
			if (dmx.pFrames->usedBuffs.get_count() > 2) {
				std::lock_guard<std::mutex> lock(dmx.displayMutex);
				VideoFrameBuffer *pFrame = &dmx.vFrameBuffer;

				glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit0"), 0);
				glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit1"), 1);
				glProgramUniform1i(shader->programId, glGetUniformLocation(shader->programId, "texUnit2"), 2);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, texIds[0]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vWidth, vHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->y);
				GL_CHECK("glGetTexImage() didn't work");

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, texIds[1]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dmx.chromaWidth, dmx.chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->u);
				GL_CHECK("glGetTexImage() didn't work");

				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, texIds[2]);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dmx.chromaWidth, dmx.chromaHeight, GL_RED, GL_UNSIGNED_BYTE, (GLvoid *)pFrame->v);
				GL_CHECK("glGetTexImage() didn't work");
				/*
				*/
			}

			XGLTexQuad::Draw();
		}
	}

	XAVDemux dmx;
};

XAVPlayer *pPlayer;
bool step;

extern bool initHmd;

void ExampleXGL::BuildScene() {

	if (false) {
		preferredSwapInterval = 0;
		preferredWidth = 1880;
		preferredHeight = 960;
		initHmd = false;
	}
	else {
		preferredSwapInterval = 0;
		preferredWidth = 1280;
		preferredHeight = 720;
	}
	
	glm::vec3 cameraPosition(0, -40, 9);
	glm::vec3 cameraDirection(0, 1, 0);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string video2Url = config.WideToBytes(config.Find(L"VideoFile2")->AsString());

	std::string videoPath;
	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	AddShape("shaders/yuv", [&](){ pPlayer = new XAVPlayer(videoPath); return pPlayer; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(-8.0f, 0.0f, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	pPlayer->model = translate * rotate *scale;

	XInputKeyFunc seekPercentFunc = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			key -= '0';
			if (key < 0)
				key = 0;
			else if (key > 9)
				key = 9;

			pPlayer->dmx.SeekPercent((float)key * 10.0f);
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
					pPlayer->dmx.playing = false;
					step = true;
					break;

				case GLFW_KEY_LEFT:
					xprintf("-1\n");
					pPlayer->dmx.playing = false;
					break;
			}
		}
	};
	AddKeyFunc(XInputKeyRange(GLFW_KEY_RIGHT, GLFW_KEY_END), seekDeltaFunc);

	AddKeyFunc('P', [&](int key, int flags){
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			pPlayer->dmx.StartPlaying();
		}
	});

	AddKeyFunc('O', [&](int key, int flags){
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat){
			pPlayer->dmx.StopPlaying();
		}
	});

	XGLGuiManager *gm = GetGuiManager();
	XGLGuiSlider *slider = (XGLGuiSlider *)(gm->FindObject("File Seek"));
	if (slider != nullptr) {
		XGLGuiCanvas::MouseEventListener mel = [slider](float x, float y, int flags) {
			if (slider->HasMouse()) {
				XGLGuiCanvas *thumb = (XGLGuiCanvas *)slider->Children()[1];
				float percent = slider->Position() * 100.0f;
				pPlayer->dmx.SeekPercent(percent);
			}
		};

		slider->AddMouseEventListener(mel);
	}

	pPlayer->StartPlaying();
}
