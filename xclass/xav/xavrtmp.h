/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAVRTMP_H
#define XAVRTMP_H

#include <rtmp.h>
#include <rtmp_log.h>
#include "xavnet.h"

class XAVRtmpThread : public XThread {
public:
	XAVRtmpThread(const std::string, XFifo *fifo);
	void *Run();

	std::string url;

	RTMP *pRtmp;
	unsigned char *buffer;
	static void logCallback(int level, const char *fmt, va_list);
	XFifo *fifo;
};

class XAVRtmp : public XAVNet {
public:
	XAVRtmp(const std::string url);
	~XAVRtmp();

private:
	XAVRtmpThread thread;
};

#endif //XAVNET_H
