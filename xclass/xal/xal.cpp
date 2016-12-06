#include "xal.h"

void CheckAlError(const char *file, int line, std::string what){
	ALenum err = alGetError();
	if (err != AL_NO_ERROR){
		std::string estr(
			file +
			std::string(":") +
			std::to_string(line) +
			std::string(": ") +
			what +
			std::string(": ") +
			std::to_string(err)
			);
		throw std::runtime_error(estr);
	}
}

XAL::XAL(ALCchar *dn, int sr, int fmt, int nb) : deviceName(dn), sampleRate(sr), format(fmt), nBuffers(nb), dqueuedIdx(0) {
	if ( (nBuffers & ~(nBuffers - 1)) != nBuffers)
		throw std::runtime_error("Number of audio buffers must be power of 2");

	EnumerateDevices();

	for (int i = 0; i < deviceList.size(); i++)
		xprintf("Device: %s\n", deviceList[i].c_str());

	if ((audioDevice = alcOpenDevice(deviceName)) == NULL)
		throw std::runtime_error("alcOpenDevice() failed");

	if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL)
		throw std::runtime_error("alcCreateContext() failed\n");

	alcMakeContextCurrent(audioContext);
	alGetError();

	alGenBuffers(nBuffers, alBufferIds);
	AL_CHECK("alGenBuffers() failed to create a buffer");

	alGenSources(1, &alSourceId);
	AL_CHECK("alGenSources() failed");
}

XAL::~XAL() {
	ALint errorCode;
	xprintf("XAL::~XAL()\n");
	alDeleteSources(1, &alSourceId);
	alDeleteBuffers(nBuffers, alBufferIds);
	errorCode = alGetError();
	alcMakeContextCurrent(NULL);
	errorCode = alGetError();
	alcDestroyContext(audioContext);
	alcCloseDevice(audioDevice);
}

void XAL::AddBuffers(int count) {
	if (count != nBuffers)
		throw std::runtime_error("AddBuffers: 'count' != 'nBuffers'");

	for (int i = 0; i < count; i++)
		shortBuffers.push_back(AudioSampleShortBuffer(audioSamples));
}

void XAL::QueueBuffers(int numSamplesToQueue, int numBuffsToQueue) {
	int i = 0;

	for (auto it : shortBuffers) {
		alBufferData(alBufferIds[i++], format, it.data(), numSamplesToQueue * sizeof(AudioSampleShort), sampleRate);
		AL_CHECK("alBufferData() failed");
		if (i == numBuffsToQueue)
			break;
	}
}

void XAL::Play() {
	alSourceQueueBuffers(alSourceId, nBuffers, alBufferIds);
	AL_CHECK("alSourceQueueBuffers() failed");

	alSourcePlay(alSourceId);
}

void XAL::Pause() {
	alSourcePause(alSourceId);
}

void XAL::Stop() {
	alSourceStop(alSourceId);
}

ALuint XAL::WaitForProcessedBuffer() {
	ALint val;
	ALuint bufferId;

	do {
		alGetSourcei(alSourceId, AL_BUFFERS_PROCESSED, &val);
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));
	} while (val == 0);

	alSourceUnqueueBuffers(alSourceId, 1, &bufferId);
	AL_CHECK("alSourceUnqueueBuffers() failed");

	// it's highly likely that this for loop will always iterate only once,
	// regardles of nBuffers value.  (which is enforced to be power-of-2)
	for (int i = 0; i < nBuffers; i++) {
		int nextIdx = (dqueuedIdx + 1 + i) & (nBuffers - 1);
		if (alBufferIds[nextIdx] == bufferId) {
			dqueuedIdx = nextIdx;
			break;
		}
	}
	return dqueuedIdx;
}

void XAL::Convert(float *left, float *right) {
	float *pLeft = left;
	float *pRight = right;

	AudioSampleShort *pass = shortBuffers[dqueuedIdx].data();

	// convert to signed short
	for (int i = 0; i < audioSamples; i++) {
		pass->left = (short)(*pLeft * 31000.0f);
		pass->right = (short)(*pRight * 31000.0f);
		pass++;
		pLeft++;
		pRight++;
	}
}

void XAL::Buffer() {
	ALuint idx = dqueuedIdx;

	alBufferData(alBufferIds[dqueuedIdx], format, shortBuffers[idx].data(), audioSamples * sizeof(AudioSampleShort), sampleRate);
	AL_CHECK("alBufferData() failed");

	alSourceQueueBuffers(alSourceId, 1, &alBufferIds[dqueuedIdx]);
	AL_CHECK("alSourceQueueBuffers() failed");

}

void XAL::Restart() {
	ALint state;

	alGetSourcei(alSourceId, AL_SOURCE_STATE, &state);
	if (state != AL_PLAYING)
		alSourcePlay(alSourceId);
}

void XAL::TestTone(int count) {
	XALShortBuffer::iterator it;
	int i=count;

	for (it = shortBuffers.begin(); (it != shortBuffers.end()); it++) {
		AudioSampleShort *s = it->data();

		for (int j = 0; j < audioSamples; j++) {
			double value = 2.0 * (double)j / (double)128 * M_PI;
			s[j].left = (short)(sin(value) * 32767.0);
			s[j].right = (short)(sin(value) * 32767.0);
		}
		i--;
		if (i == 0)
			break;
	}
}

XALDeviceList XAL::EnumerateDevices() {
	ALboolean hasEnumerate;

	if (deviceList.size() == 0) {
		hasEnumerate = alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT");
		if (hasEnumerate == AL_TRUE) {
			const ALCchar *device = alcGetString(NULL, ALC_ALL_DEVICES_SPECIFIER);
			while (device && *device != '\0') {
				deviceList.push_back(std::string(device));
				device += strlen(device) + 1;
			}
		}
	}

	return deviceList;
}