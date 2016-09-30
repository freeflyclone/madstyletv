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

#define AUDIO_SAMPLES 1024
#define AL_CHECK(what) CheckError(__FILE__,__LINE__,what)

typedef struct {
	float left;
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

	short testToneBuffer[48000 * 4];
};

#endif