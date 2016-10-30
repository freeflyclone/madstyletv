#include "xavenc.h"
#include "xutils.h"

std::map<std::wstring, AVPixelFormat> pixelFormats = { 
	{ L"AV_PIX_FMT_YUV420P", AV_PIX_FMT_YUV420P },
	{ L"AV_PIX_FMT_YUV422P", AV_PIX_FMT_YUV422P },
	{ L"AV_PIX_FMT_YUV444P", AV_PIX_FMT_YUV444P }
};

XAVEncoder::XAVEncoder(XConfig *cfg, unsigned char *y, unsigned char *u, unsigned char *v) : config(cfg), yBuffer(y), uBuffer(u), vBuffer(v), frameNumber(0), output(NULL), udpSocket(0) {
	avcodec_register_all();
	char udpAddress[17] = { "224.1.1.1" };
	int udpPort = 5555;

	if ((codec = avcodec_find_encoder(AV_CODEC_ID_H264)) == NULL)
		throw std::runtime_error("couldn't find H264 encoder");

	if ((ctx = avcodec_alloc_context3(codec)) == NULL)
		throw std::runtime_error("avcodec_alloc_context3() failed");

	if (config->Find(L"Encoder")) {
		ctx->bit_rate = (int)config->Find(L"Encoder.bitrate")->AsNumber();
		ctx->width = (int)config->Find(L"Encoder.width")->AsNumber();
		ctx->height = (int)config->Find(L"Encoder.height")->AsNumber();
		ctx->gop_size = (int)config->Find(L"Encoder.gopsize")->AsNumber();
		ctx->max_b_frames = (int)config->Find(L"Encoder.max_b_frames")->AsNumber();
		ctx->time_base = {
			(int)config->Find(L"Encoder.timebase")->AsArray()[0]->AsNumber(),
			(int)config->Find(L"Encoder.timebase")->AsArray()[1]->AsNumber()
		};
		ctx->pix_fmt = pixelFormats[config->Find(L"Encoder.pixelFormat")->AsString()];
		av_opt_set(ctx->priv_data, "preset", config->WideToBytes(config->Find(L"Encoder.preset")->AsString()).c_str(), 0);

		std::string tmp = config->WideToBytes(config->Find(L"Encoder.udpaddress")->AsString());
		strncpy(udpAddress, tmp.c_str(), sizeof(udpAddress));
		udpPort = (int)config->Find(L"Encoder.udpport")->AsNumber();
	}
	else {
		xprintf("Oops, no \"Encoder\" found in config file, using defaults\n");
		ctx->bit_rate = 8000000;
		ctx->width = 1280;
		ctx->height = 720;
		ctx->gop_size = 20;
		ctx->max_b_frames = 1;
		ctx->time_base = { 1, 60 };
		ctx->pix_fmt = AV_PIX_FMT_YUV444P;
		av_opt_set(ctx->priv_data, "preset", "fast", 0);
	}

	if (avcodec_open2(ctx, codec, NULL) < 0)
		throw std::runtime_error("avcodec_open2() failed");

	if ((frame = av_frame_alloc()) == NULL)
		throw std::runtime_error("av_frame_alloc() failed");

	frame->format = ctx->pix_fmt;
	frame->width = ctx->width;
	frame->height = ctx->height;

	if (av_image_alloc(frame->data, frame->linesize, ctx->width, ctx->height, ctx->pix_fmt, 32) < 0)
		throw std::runtime_error("av_image_alloc() failed");

	if ((output = fopen("test.m4v", "wb")) == NULL)
		throw std::runtime_error("failed to create output file");

	SocketsSetup();

	if ((udpSocket = SocketOpen(NULL, udpPort, SOCK_DGRAM, IPPROTO_UDP, 0)) <= 0)
		xprintf("Failed to open udpSocket\n");

	// UDP requires sendto(2), which requires a destination address
	SockAddrIN(&udpDest, (char *)udpAddress, udpPort);
}

XAVEncoder::~XAVEncoder() {
	xprintf("XAVEncoder::~XAVEncoder()\n");

	avcodec_close(ctx);
	av_free(ctx);
	av_freep(&frame->data[0]);
	av_frame_free(&frame);

	if (output)
		fclose(output);
}

void XAVEncoder::SetParams(void *p){
	xprintf("XAVEncoder::SetParams()\n");
}

void XAVEncoder::EncodeFrame(unsigned char *img, int width, int height, int depth){
	int ret,gotOutput;

	frame->pts = frameNumber++;

	frame->data[0] = yBuffer;
	frame->data[1] = uBuffer;
	frame->data[2] = vBuffer;

	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;

	if ((ret = avcodec_encode_video2(ctx, &pkt, frame, &gotOutput)) < 0)
		throw std::runtime_error("avcodec_encode_video2() failed");

	if (gotOutput) {
		fwrite(pkt.data, 1, pkt.size, output);
		fflush(output);

		sendto(udpSocket, (const char *)pkt.data, pkt.size, 0, (const struct sockaddr *)&udpDest, sizeof(udpDest));
	}
}