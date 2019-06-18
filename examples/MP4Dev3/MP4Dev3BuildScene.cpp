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

static const int vWidth = 1920;
static const int vHeight = 1080;

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
	XAVDemux(std::string fn, XGLContextImage* p) : fileName(fn), pci(p), XThread("XAVDemux") {
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

		// get AVCodecContext for video stream
		pCodecCtx = pFormatCtx->streams[vStreamIdx]->codec;

		// xyzzy: for now, assume my machine (Windows 7: quad core w/HT)
		// apparently old FFMPEG did this automagically, whereas new does not
		if (true) {
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
		// make sure we're using XGLContextImage's secondary OpenGL context
		pci->MakeContextCurrent();

		while (IsRunning()) {
			if (playing) {
				std::lock_guard<std::mutex> lock(playMutex);
				if ((retVal = av_read_frame(pFormatCtx, &packet)) == 0) {
					if (packet.stream_index == vStreamIdx) {
						int frameFinished;

						avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

						if (frameFinished)
							pci->UploadToTexture(pFrame->data[0], pFrame->data[1], pFrame->data[2]);
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
		StopPlaying();
		ended = true;
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

	XGLContextImage* pci{ nullptr };
	XTimer xtDecoder;
};

class XAVPlayer : public XGLContextImage {
public:
	XAVPlayer(ExampleXGL *pxgl, std::string url) : dmx(url, this), XGLContextImage(pxgl, vWidth, vHeight) {}

	~XAVPlayer() {
		dmx.WaitForStop();
	}

	void StartPlaying() {
		if (dmx.IsRunning() == false)
			dmx.Start();

		dmx.StartPlaying();
	}

	void StopPlaying() {
		dmx.StopPlaying();
	}

	XAVDemux dmx;
};

XAVPlayer *pPlayer, *pPlayer2, *pPlayer3;
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
	
	glm::vec3 cameraPosition(0, -21, 9);
	glm::vec3 cameraDirection(0, 1, 0);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string video2Url = config.WideToBytes(config.Find(L"VideoFile2")->AsString());
	std::string video3Url = config.WideToBytes(config.Find(L"VideoFile3")->AsString());

	std::string videoPath, video2Path, video3Path;

	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	if (video2Url.find("http") != video2Url.npos)
		video2Path = video2Url;
	else if (video2Url.find(":", 1) != video2Url.npos)
		video2Path = video2Url;
	else
		video2Path = pathToAssets + "/" + video2Url;

	if (video3Url.find("http") != video3Url.npos)
		video3Path = video3Url;
	else if (video3Url.find(":", 1) != video3Url.npos)
		video3Path = video3Url;
	else
		video3Path = pathToAssets + "/" + video2Url;

	AddShape("shaders/yuv", [&](){ pPlayer = new XAVPlayer(this, videoPath); return pPlayer; });
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	pPlayer->model = translate * rotate *scale;

	//AddShape("shaders/yuv", [&](){ pPlayer2 = new XAVPlayer(this, video2Path); return pPlayer2; });
	//translate = glm::translate(glm::mat4(), glm::vec3(16.0f, 0.0f, 9.0f));
	//pPlayer2->model = translate * rotate *scale;

	//AddShape("shaders/yuv", [&](){ pPlayer3 = new XAVPlayer(this, video3Path); return pPlayer3; });
	//translate = glm::translate(glm::mat4(), glm::vec3(-16.0f, 0.0f, 27.0f));
	//pPlayer3->model = translate * rotate *scale;

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

	XGLGuiSlider *fpsSlider = (XGLGuiSlider *)(gm->FindObject("Frames/Second"));
	if (fpsSlider != nullptr) {
		XGLGuiCanvas::MouseEventListener mel = [fpsSlider](float x, float y, int flags) {
			if (fpsSlider->HasMouse()) {
				XGLGuiCanvas *thumb = (XGLGuiCanvas *)fpsSlider->Children()[1];
				float percent = fpsSlider->Position() * 400.0f + 1;
				pPlayer->SetStepFrequency((int)percent);
			}
		};

		fpsSlider->AddMouseEventListener(mel);
	}

	pPlayer->StartPlaying();
	//pPlayer2->StartPlaying();
	//pPlayer3->StartPlaying();
}
