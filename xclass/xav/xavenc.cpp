#include "xutils.h"
#include "xavenc.h"

XAVEncoder::XAVEncoder() : frameNumber(0), output(NULL) {
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

	if ((convertCtx = sws_getContext(ctx->width, ctx->height, AV_PIX_FMT_BGR24, ctx->width, ctx->height, AV_PIX_FMT_YUV420P, SWS_POINT, NULL, NULL, NULL)) == NULL)
		throw std::runtime_error("sws_getContext() failed");

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
	unsigned char *src_planes[3] = { img, NULL, NULL };
	unsigned char *dest_planes[3] = {frame->data[0], frame->data[1], frame->data[2]};
	int src_stride[3] = { width * 3, 0, 0 };
	int dest_stride[3] = { width, width / 2, width / 2 };
	int ret,gotOutput;

	sws_scale(convertCtx, src_planes, src_stride, 0, height, dest_planes, dest_stride);

	frame->pts = frameNumber++;

	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;

	if ((ret = avcodec_encode_video2(ctx, &pkt, frame, &gotOutput)) < 0)
		throw std::runtime_error("avcodec_encode_video2() failed");

	if (gotOutput) {
		xprintf("Got output: %d\n", pkt.size);
		fwrite(pkt.data, 1, pkt.size, output);
		fflush(output);
	}
}