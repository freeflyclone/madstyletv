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
#include "xglcontextimage.h"

//static const int numFrames = 4;
static const int vWidth = 1920;
static const int vHeight = 1080;

uint8_t* pGlobalPboBuffer = nullptr;

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
	XAVDemux(std::string fn) : fileName(fn), XThread("XAVDemux") {
		av_register_all();

		ended = true;
		StartPlaying();
	}

	~XAVDemux() {
		xprintf("%s()\n", __FUNCTION__);
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
		int ySize = frame->width * frame->height;
		int uvSize = ySize / 4;

		uint8_t* y = pGlobalPboBuffer + INDEX(self->nDecoded++) * (ySize + (uvSize*2));
		uint8_t* u = y + ySize;
		uint8_t* v = u + uvSize;

		if (y) {
			frame->data[0] = y;
			frame->data[1] = u;
			frame->data[2] = v;

			frame->linesize[0] = s->width;
			frame->linesize[1] = self->chromaWidth;
			frame->linesize[2] = self->chromaWidth;

			frame->buf[0] = av_buffer_create(frame->data[0], frame->linesize[0] * frame->height, our_buffer_default_free, self, 0);
			frame->buf[1] = av_buffer_create(frame->data[1], frame->linesize[1] * frame->height / 2, our_buffer_default_free, self, 0);
			frame->buf[2] = av_buffer_create(frame->data[2], frame->linesize[2] * frame->height / 2, our_buffer_default_free, self, 0);
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

		// xyzzy: for now, assume my machine (Windows 7: quad core w/HT)
		// apparently old FFMPEG did this automagically, whereas new does not
		if (false) {
			pCodecCtx->thread_count = 4;
			pCodecCtx->thread_type = FF_THREAD_FRAME;
		}

		if ((pCodec = avcodec_find_decoder(pCodecCtx->codec_id)) == nullptr)
			throwXAVException("Unsupported codec " + fileName + "!\n");

		if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
			throwXAVException("Unable to open codec " + fileName + "");

		// save timing information (so we stand a chance of appropriate playback speed)
		timeBase = pCodecCtx->time_base;
		ticksPerFrame = pCodecCtx->ticks_per_frame;

		xprintf("video context delay: %d, num: %d, den: %d\n", pCodecCtx->delay, timeBase.num, timeBase.den);

		// figure out the chroma sub-sampling of the pixel format
		AVPixelFormat pixelFormat = pCodecCtx->pix_fmt;
		const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get(pixelFormat);

		XGLPixelFormatDescriptor* xpfd = new XGLPixelFormatDescriptor(pixelFormat);

		// save it so our clients can know
		chromaWidth = pCodecCtx->width / (1 << pixDesc->log2_chroma_w);
		chromaHeight = pCodecCtx->height / (1 << pixDesc->log2_chroma_h);
	}

	void ReleaseAllTheThings(){
		xprintf("%s()\n", __FUNCTION__);

		if (pCodecCtx) {
			avcodec_close(pCodecCtx);
			pCodecCtx = nullptr;
		}

		avformat_close_input(&pFormatCtx);

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
			if (playing) {
				std::lock_guard<std::mutex> lock(playMutex);
				if ((retVal = av_read_frame(pFormatCtx, &packet)) == 0) {
					if (packet.stream_index == vStreamIdx) {
						// copy the XAVPacket
						vPkt = packet;
						int frameFinished;
						// since we have our own AVFrame allocator we know when a new AVFrame
						// is made, so we just need to run the decoder.
						avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &vPkt);
						if (frameFinished) {
							if (pGlobalPboBuffer)
								xprintf("frame finished(): %d\n", nDecoded);
						}
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
		xprintf("%s() exiting\n", __FUNCTION__);
	}

private:
	void Seek(int64_t timeOffset) {
		std::lock_guard<std::mutex> lock(playMutex);
		wasPlaying = playing;
		playing = false;

		retVal = avformat_seek_file(pFormatCtx, -1, INT64_MIN, timeOffset, INT64_MAX, 0);
		avcodec_flush_buffers(pCodecCtx);
		showFrameStatus = true;
		if (wasPlaying)
			playing = true;
	}

public:
	void SeekPercent(float percent) {
		xprintf("%s(): seeking to: %0.4f%%\n", __FUNCTION__, percent);
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
		}
		playing = true;
	}

	void StopPlaying() {
		xprintf("%s()\n", __FUNCTION__);
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

	int	chromaWidth{ 0 }, chromaHeight{ 0 };
	bool playing{ false };
	bool ended{ false };
	bool wasPlaying{ false };
	bool showFrameStatus{ false };
	float currentPlayTime{ 0.0 };

	std::mutex playMutex;
	std::mutex displayMutex;

	uint64_t nDecoded{ 0 };
};

class XAVPlayer : public XGLContextImage {
public:
	XAVPlayer(ExampleXGL *pxgl, std::string url) : dmx(url), XGLContextImage(pxgl, vWidth, vHeight, 1) {
		xprintf("%s()\n", __FUNCTION__);
	}

	~XAVPlayer() {
		dmx.WaitForStop();
	}

	void StartPlaying() {
		if (dmx.IsRunning() == false)
			dmx.Start();

		dmx.StartPlaying();
		Start();
	}

	void StopPlaying() {
		Stop();
		dmx.StopPlaying();
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
		preferredSwapInterval = 1;
		preferredWidth = 1280;
		preferredHeight = 720;
	}
	
	glm::vec3 cameraPosition(0, -21, 9);
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

	AddShape("shaders/yuv", [&](){ pPlayer = new XAVPlayer(this, videoPath); return pPlayer; });

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 9.0f));
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
