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
#include "xcircularbuffer.h"
#include "xavexcept.h"

class XAVStream;

typedef std::shared_ptr<XAVStream> XAVStreamHandle;
typedef std::vector<XAVStreamHandle> XAVStreamSet;

typedef std::function<void(uint8_t*, size_t, uint64_t)> XAVDataFunction;
typedef std::vector<XAVDataFunction> XAVDataFunctions;

// Multimedia sources possibly have more than one "stream" (audio/video for ex.)
class XAVStream
{
public:
	static const int numFrames = 4;
	static const int maxChannels = 8;

	typedef struct {
		unsigned char *buffers[maxChannels];
		int nChannels;
		int size;
		int count;
	} XAVBuffer;

	XAVStream(AVCodecContext *context);
	bool Decode(AVPacket *packet);
	void AllocateBufferPool(int number, int size, int channels);
	XAVBuffer GetBuffer();
	void ReleaseBuffer();

	void Acquire();
	void Release();

	/*
	void AddDataFunction(XAVDataFunction fn) { dataFunctions.emplace_back(fn); }

	void InvokeDataFunctions(uint8_t *data, size_t size, uint64_t time) {
		for (auto fn : dataFunctions) {
			fn(data, size, time);
		}
		totalBytes += size;
	}
	*/

	int nFramesDecoded;
	int nFramesRead;
	
	// number of audio channels (audio streams only)
	int channels;

	// size of a single channel sample in bytes
	int formatSize;

	// is this a floating point format?
	bool isFloat;
	
	int sampleRate;

	int streamIdx;

	int width, height;
	int chromaWidth, chromaHeight;
	double streamTime;

//private:
	AVCodecContext *pCodecCtx;
	AVCodec *pCodec;
	AVFrame *pFrame;
	AVStream *pStream;

	// buffer pool for decoded frames.
	XAVBuffer frames[numFrames];

	//unsigned char *buffer;
	int numBytes;
	int frameFinished;
	XSemaphore freeBuffs;
	XSemaphore usedBuffs;

	//int64_t totalBytes, totalChunks;
	//XAVDataFunctions dataFunctions;

	XCircularBuffer *pcb;
};

class XAVSrc : public XThread
{
public:
	XAVSrc(const std::string name, bool v, bool a);
	XAVSrc();
	bool DecodeVideo(AVPacket *packet);
	bool DecodeAudio(AVPacket *packet);
	virtual void Run();

	XAVStream *VideoStream();
	XAVStream *AudioStream();

	// these need to be public for derived classes.
	AVFormatContext *pFormatCtx;
	int mNumStreams;
	int mUsedStreams;
	AVPacket packet;
	XAVStreamSet mStreams;
	XAVStreamHandle mVideoStream;
	XAVStreamHandle mAudioStream;
	std::string name;

	bool doVideo, doAudio;
};

#endif // XAVSRC_H
