/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of XAVEncoder - an interface to x264 encoder library:
****************************************************************************/
#ifndef XAVENC_H
#define XAVENC_H
#include "socket.h"
#include "xshmem.h"
#include "xconfig.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
}

class XAVEncoder {
public:
	XAVEncoder(XConfig *cfg, unsigned char *y, unsigned char *u, unsigned char *v);
	~XAVEncoder();

	void SetParams(void *params);

	// encode an RGB frame, doing RGB -> YUV conversion
	void EncodeFrame(unsigned char *img, int width, int height, int depth);

	XConfig *config;

	AVCodec *codec;
	AVCodecContext *ctx;
	AVFrame *frame;
	AVPacket pkt;

	unsigned char *yBuffer, *uBuffer, *vBuffer;

	int frameNumber;
	FILE *output;
	SOCKET udpSocket;
	SOCKADDR_IN udpDest;
};

#endif