/**************************************************************
** MP4DevBuildScene.cpp
**
** Minimal libavcodec/libavformat demo, capable of decoding
** & rendering GoPro files shot in HD @ 120 FPS
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
	MP4Demux(const char *filename) : XThread("MP4Demux") {
		av_register_all();

		if ((pFormatCtx = avformat_alloc_context()) == nullptr)
			throwXAVException("Unable to allocate AVFormatContext");

		if ((pFrame = avcodec_alloc_frame()) == nullptr)
			throwXAVException("Unable to allocate AVFrame");

		if (avformat_open_input(&pFormatCtx, filename, 0, NULL) != 0)
			throwXAVException("avformat_open_input failed: Couldn't open file\n");

		if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
			throwXAVException("avformat_find_stream_info failed: Couldn't find stream information\n");

		for (int i = 0; pFormatCtx->nb_streams; i++) {
			if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) { //CODEC_TYPE_VIDEO
				vStreamIdx = i;
				break;
			}
		}

		if (vStreamIdx == -1)
			throwXAVException("No video stream found in " + std::string(filename));

		pCodecCtx = pFormatCtx->streams[vStreamIdx]->codec;

		if ((pCodec = avcodec_find_decoder(pCodecCtx->codec_id)) == nullptr)
			throwXAVException("Unsupported codec!\n");

		if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
			throwXAVException("Unable to open codec");

		// figure out the chroma sub-sampling of the pixel format
		int pixelFormat = pCodecCtx->pix_fmt;
		const AVPixFmtDescriptor *pixDesc = av_pix_fmt_desc_get((AVPixelFormat)pixelFormat);

		// save it so our clients can know
		chromaWidth = pCodecCtx->width / (1 << pixDesc->log2_chroma_w);
		chromaHeight = pCodecCtx->height / (1 << pixDesc->log2_chroma_h);

		pFrames = new FramePool(pCodecCtx->width, pCodecCtx->height);
	}

	~MP4Demux() {
		avcodec_close(pCodecCtx);
		avformat_close_input(&pFormatCtx);
	}

	void Run() {
		while (IsRunning() && (retVal == 0)) {
			if ((retVal = av_read_frame(pFormatCtx, &packet)) == 0) {
				if (packet.stream_index == vStreamIdx) {
					int frameFinished;
					avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);
					if (frameFinished) {
						if (pFrames->freeBuffs.wait_for(1000)) {
							VideoFrame *pvf = pFrames->frames[nFramesDecoded & (numFrames - 1)];

							int ySize = pFrame->height * pFrame->linesize[0];
							int uvSize = chromaWidth * chromaHeight;

							memcpy(pvf->y, pFrame->data[0], ySize);
							memcpy(pvf->u, pFrame->data[1], uvSize);
							memcpy(pvf->v, pFrame->data[2], uvSize);

							nFramesDecoded++;
							pFrames->usedBuffs.notify();
						}
						else {
							xprintf("Timed out waiting for freeBuff: %d\n", pFrames->freeBuffs.get_count());
						}
						av_frame_unref(pFrame);
					}
				}
				av_free_packet(&packet);
			}
		}
	}

	AVFormatContext *pFormatCtx;
	int             vStreamIdx{ -1 };
	AVCodecContext  *pCodecCtx;
	AVCodec         *pCodec;
	AVPacket		packet;
	AVFrame			*pFrame;
	int retVal{ 0 };

	FramePool *pFrames;
	int nFramesDecoded{ 0 };
	int nFramesDisplayed{ 0 };
	int chromaWidth, chromaHeight;
};

MP4Demux *pMp4;

void ExampleXGL::BuildScene() {
	preferredSwapInterval = 1;

	std::string videoUrl = config.WideToBytes(config.Find(L"VideoFile")->AsString());
	std::string videoPath;
	if (videoUrl.find("http") != videoUrl.npos)
		videoPath = videoUrl;
	else if (videoUrl.find(":", 1) != videoUrl.npos)
		videoPath = videoUrl;
	else
		videoPath = pathToAssets + "/" + videoUrl;

	pMp4 = new MP4Demux(videoPath.c_str());

	XGLShape *shape;
	AddShape("shaders/yuv", [&](){ shape = new XGLTexQuad(vWidth, vHeight, 1); return shape; });
	shape->AddTexture(vWidth / 2, vHeight / 2, 1);
	shape->AddTexture(vWidth / 2, vHeight / 2, 1);

	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(16.0f, 9.0f, 1.0f));
	glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 9.0f));
	glm::mat4 rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	shape->model = translate * rotate *scale;

	shape->SetAnimationFunction([&, shape](float clock) {
		static float oldClock = 0.0f;
		//if (clock > oldClock) {
			oldClock = clock;
			if ((pMp4->nFramesDecoded - pMp4->nFramesDisplayed) > (numFrames / 2)) {
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
		//}
	});

	pMp4->Start();
}
