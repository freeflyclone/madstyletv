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

typedef struct {
	float left;
	float right;
} AudioSampleFloat, AudioSampleFloatBuffer[AUDIO_SAMPLES];

typedef struct {
	short left;
	short right;
} AudioSampleShort, AudioSampleShortBuffer[AUDIO_SAMPLES];

typedef std::vector<std::string> XALDeviceList;
typedef std::vector<AudioSampleFloatBuffer> XALFloatBuffer;
typedef std::vector<AudioSampleShortBuffer> XALShortBuffer;

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
	int sampleRate;

	short testToneBuffer[48000 * 4];
};

#endif
