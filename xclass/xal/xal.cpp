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

XAL::XAL(ALCchar *dn, int sr, int fmt, int nb) : deviceName(dn), sampleRate(sr), format(fmt), nBuffers(nb) {
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
		shortBuffers.push_back(AudioSampleShortBuffer(AUDIO_SAMPLES));
}

void XAL::QueueBuffers() {
	XALShortBuffer::iterator it;
	int i = 0;

	for (it = shortBuffers.begin(); it != shortBuffers.end(); it++) {
		alBufferData(alBufferIds[i++], format, it->data(), AUDIO_SAMPLES*sizeof(AudioSampleShort), sampleRate);
		AL_CHECK("alBufferData() failed");
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

void XAL::TestTone(int count) {
	XALShortBuffer::iterator it;
	int i;

	for (i=0, it = shortBuffers.begin(); (it != shortBuffers.end()) && (i<count); it++,i++) {
		AudioSampleShort *s = it->data();

		for (int j = 0; j < AUDIO_SAMPLES; j++) {
			double value = 2.0 * (double)j / (double)128 * M_PI;
			s[j].left = (short)(sin(value) * 32767.0);
			s[j].right = (short)(sin(value) * 32767.0);
		}
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