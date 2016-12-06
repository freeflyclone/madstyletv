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
#include <chrono>
#include <ctime>
#include <thread>

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

typedef struct {
	short left;
	short right;
} AudioSampleShort;

typedef std::vector<AudioSampleShort> AudioSampleShortBuffer;
typedef std::vector<AudioSampleShortBuffer> XALShortBuffer;

typedef std::vector<std::string> XALDeviceList;

class XAL {
public:
	XAL(ALCchar *dn = (NULL), int sr = defaultSamplerate, int fmt = defaultFormat, int nb=1);
	virtual ~XAL();

	void AddBuffers(int count);
	void QueueBuffers(int spq = audioSamples, int ntq = maxBuffers);
	void Play();
	void Pause();
	void Stop();

	ALuint WaitForProcessedBuffer();
	void Convert(float *, float *);
	void Buffer();
	void Restart();

	void TestTone(int count);

	XALDeviceList EnumerateDevices();

	// these must both be powers of 2
	static const int audioSamples = 1024;
	static const int maxBuffers = 64;

	static const int defaultSamplerate = 48000;
	static const int defaultFormat = AL_FORMAT_STEREO16;
private:
	XALDeviceList deviceList;

	ALCchar *deviceName;
	ALCdevice *audioDevice;
	ALCcontext *audioContext;
	ALuint alBufferIds[maxBuffers];
	ALuint alSourceId;
	ALenum alError;
	int sampleRate;
	int format;
	int nBuffers;
	XALShortBuffer shortBuffers;
	ALuint dqueuedIdx;
};

#endif
