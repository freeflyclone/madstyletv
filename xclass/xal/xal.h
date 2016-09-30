/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAL_H
#define XAL_H

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <stdexcept>

// OpenAL includes
#include <al.h>
#include <alc.h>

#include "xutils.h"

typedef std::vector<std::string> XALDeviceList;

class XAL {
public:
	XAL();
	virtual ~XAL();

	void Play();
	void Pause();
	void Stop();

	void TestTone();

	XALDeviceList EnumerateDevices();

private:
	XALDeviceList deviceList;

	ALCchar *deviceName;
	ALCdevice *audioDevice;
	ALCcontext *audioContext;
	ALuint alBufferId;
	ALuint alSourceId;
	ALenum alError;

	short testToneBuffer[48000 * 4];
};

#endif