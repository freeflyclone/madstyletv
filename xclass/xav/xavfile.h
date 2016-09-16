/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAVFILE_H
#define XAVFILE_H

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
#include "xutils.h"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
};

#include "xthread.h"
#include "xavexcept.h"
#include "xavsrc.h"

class XAVFile : public XAVSrc {
public:
	XAVFile(const std::string url);
	~XAVFile();

	int Read(uint8_t *buff, int size);
	int Write(uint8_t *buff, int size);
	int64_t Seek(int64_t offset, int whence);

	std::string url;

private:
	static int read(void *opaque, uint8_t *buff, int size);
	static int write(void *opaque, uint8_t *buff, int size);
	static int64_t seek(void *opaque, int64_t offset, int whence);
};

#endif //XAVFILE_H
