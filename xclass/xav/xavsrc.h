/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAVSRC_H
#define XAVSRC_H

#ifdef WIN32
#include <windows.h>
#include <winsock.h>
#endif

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <stdexcept>


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include "xutils.h"
};

#include "xthread.h"
#include "xavexcept.h"

// Multimedia sources possibly have more than one "stream" (audio/video for ex.)
class XAVStream
{
public:
	XAVStream(AVCodecContext *context);
	bool Decode(AVPacket *packet);
	unsigned char *GetBuffer();
	void ReleaseBuffer();

	void Acquire();
	void Release();
private:
	AVCodecContext *pCodecCtx;
	AVCodec *pCodec;
	AVFrame *pFrame;

	unsigned char *buffer;
	int numBytes;
	int frameFinished;
	int nFramesDecoded;
	XSemaphore freeBuffs;
	XSemaphore usedBuffs;
};

class XAVSrc : public XThread
{
public:
	XAVSrc(const std::string name);
	XAVSrc();
	bool DecodeVideo(AVPacket *packet);
	bool DecodeAudio(AVPacket *packet);
	virtual void Run();

	XAVStream *VideoStream();
	XAVStream *AudioStream();

	// these need to be public for derived classes.
	AVFormatContext *pFormatCtx;
	int mNumStreams;
	AVPacket packet;
	std::vector<std::shared_ptr<XAVStream> >mStreams;
	std::shared_ptr<XAVStream> mVideoStream;
	std::shared_ptr<XAVStream> mAudioStream;
	std::string name;
};

#endif // XAVSRC_H
