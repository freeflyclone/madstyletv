/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAVNET_H
#define XAVNET_H

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
#include "utils.h"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
};

#include "xclasses.h"
#include "xavexcept.h"
#include "xavsrc.h"

#define qDebug DebugPrintf

class XAVNet : public XAVSrc {
public:
	XAVNet(const std::string url);
	~XAVNet();
	virtual void *Run();

	int Read(uint8_t *buff, int size);
	int Write(uint8_t *buff, int size);
	int64_t Seek(int64_t offset, int whence);

	unsigned char *buffer;
	unsigned char detectBuffer[65536];
	XFifo fifo;
	std::string url;
	AVIOContext *pAvioCtx;

private:
	static int read(void *opaque, uint8_t *buff, int size);
	static int write(void *opaque, uint8_t *buff, int size);
	static int64_t seek(void *opaque, int64_t offset, int whence);
	
};

#endif //XAVNET_H
