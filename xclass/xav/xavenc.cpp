#include "xutils.h"
#include "xavenc.h"

XAVEncoder::XAVEncoder(){
	xprintf("XAVEncoder::XAVEncoder()\n");

	avcodec_register_all();

	if ((codec = avcodec_find_encoder(AV_CODEC_ID_H264)) == NULL)
		throw std::runtime_error("couldn't find H264 encoder");

	if ((ctx = avcodec_alloc_context3(codec)) == NULL)
		throw std::runtime_error("avcodec_alloc_context3() failed");

	ctx->bit_rate = 4000000;
	ctx->width = 1280;
	ctx->height = 720;
	ctx->time_base = { 1, 60 };
	ctx->gop_size = 10;
	ctx->max_b_frames = 1;
	ctx->pix_fmt = AV_PIX_FMT_YUV420P;

	av_opt_set(ctx->priv_data, "preset", "ultrafast", 0);

	if (avcodec_open2(ctx, codec, NULL) < 0)
		throw std::runtime_error("avcodec_open2() failed");

	if ((frame = av_frame_alloc()) == NULL)
		throw std::runtime_error("av_frame_alloc() failed");

	frame->format = ctx->pix_fmt;
	frame->width = ctx->width;
	frame->height = ctx->height;

	if (av_image_alloc(frame->data, frame->linesize, ctx->width, ctx->height, ctx->pix_fmt, 32) < 0)
		throw std::runtime_error("av_image_alloc() failed");
}

XAVEncoder::~XAVEncoder() {
	xprintf("XAVEncoder::~XAVEncoder()\n");

	avcodec_close(ctx);
	av_free(ctx);
	av_freep(&frame->data[0]);
	av_frame_free(&frame);
}

void XAVEncoder::SetParams(void *p){
	xprintf("XAVEncoder::SetParams()\n");
}

void XAVEncoder::EncodeFrame(unsigned char *img, int width, int height, int depth){
	xprintf("XAVEncoder::EncodeFrame\n");
}