/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of XAVEncoder - an interface to x264 encoder library:
****************************************************************************/
#ifndef XAVENC_H
#define XAVENC_H
#include "xshmem.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
}

class XAVEncoder {
public:
	XAVEncoder();
	~XAVEncoder();

	void SetParams(void *params);

	// encode an RGB frame, doing RGB -> YUV conversion
	void EncodeFrame(unsigned char *img, int width, int height, int depth);

	AVCodec *codec;
	AVCodecContext *ctx;
	AVFrame *frame;
	AVPacket pkt;

	SwsContext *convertCtx;
};

#endif