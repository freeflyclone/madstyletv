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

#include <string.h>
#include <math.h>

// OpenAL includes
#ifdef _APPLE_
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#else
#include <al.h>
#include <alc.h>
#endif

#include "xutils.h"

void CheckAlError(const char *, int, std::string);
#define AL_CHECK(what) CheckAlError(__FILE__,__LINE__,what)

#define AUDIO_SAMPLES 1024
#define XAL_MAX_BUFFERS	48

typedef struct {
	short left;
	short right;
} AudioSampleShort;

typedef std::vector<AudioSampleShort> AudioSampleShortBuffer;
typedef std::vector<AudioSampleShortBuffer> XALShortBuffer;

typedef std::vector<std::string> XALDeviceList;

class XAL {
public:
	XAL(ALCchar *dn = (NULL), int sr = 48000, int fmt = AL_FORMAT_STEREO16, int nb=1);
	virtual ~XAL();

	void AddBuffers(int count);
	void QueueBuffers();
	void Play();
	void Pause();
	void Stop();

	void TestTone(int count);

	XALDeviceList EnumerateDevices();

private:
	XALDeviceList deviceList;

	ALCchar *deviceName;
	ALCdevice *audioDevice;
	ALCcontext *audioContext;
	ALuint alBufferIds[XAL_MAX_BUFFERS];
	ALuint alSourceId;
	ALenum alError;
	int sampleRate;
	int format;
	int nBuffers;
	XALShortBuffer shortBuffers;
};

#endif
