#include "xutils.h"
#include "xavenc.h"

XAVEncoder::XAVEncoder(unsigned char *y, unsigned char *u, unsigned char *v) : yBuffer(y), uBuffer(u), vBuffer(v), frameNumber(0), output(NULL) {
	xprintf("XAVEncoder::XAVEncoder()\n");

	avcodec_register_all();

	if ((codec = avcodec_find_encoder(AV_CODEC_ID_H264)) == NULL)
		throw std::runtime_error("couldn't find H264 encoder");

	if ((ctx = avcodec_alloc_context3(codec)) == NULL)
		throw std::runtime_error("avcodec_alloc_context3() failed");

	ctx->bit_rate = 12000000;
	ctx->width = 1280;
	ctx->height = 720;
	ctx->time_base = { 1, 60 };
	ctx->gop_size = 10;
	ctx->max_b_frames = 1;
	// do 4:4:4 for now, until I figure out how to sub-sample U and V
	ctx->pix_fmt = AV_PIX_FMT_YUV444P;

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

	if ((output = fopen("test.m4v", "wb")) == NULL)
		throw std::runtime_error("failed to create output file");
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
	}
}