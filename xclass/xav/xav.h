/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAV_H
#define XAV_H

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <stdexcept>


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include "xutils.h"
};

//#include <rtmp.h>
//#include <rtmp_log.h>
#include "xthread.h"
#include "xavexcept.h"
#include "xavsrc.h"

// An entire presentation has a worker thread to manage scheduling of
// real-time events.
class XAV : public XThread
{
public:
	XAV();
	void AddSrc(const std::shared_ptr<XAVSrc> src);
	void Run();
	std::shared_ptr<XAVSrc>GetSrc(int idx);

	std::vector<std::shared_ptr<XAVSrc>> mSrcs;
	WSADATA wsaData;
};


#endif // AVWRAP_H
