#include "xal.h"

XAL::XAL() : deviceName(NULL) {
	EnumerateDevices();

	for (int i = 0; i < deviceList.size(); i++)
		xprintf("Device: %s\n", deviceList[i].c_str());

	if ((audioDevice = alcOpenDevice(deviceName)) == NULL) {
		throw std::runtime_error("alcOpenDevice() failed");
	}

	if ((audioContext = alcCreateContext(audioDevice, NULL)) == NULL) {
		throw std::runtime_error("alcCreateContext() failed\n");
	}

	alcMakeContextCurrent(audioContext);
	alGetError();

	alGenBuffers(1, &alBufferId);
	if ((alError = alGetError()) != AL_NO_ERROR)
		throw std::runtime_error("alGenBuffers() failed to create a buffer");

	alGenSources(1, &alSourceId);
	if ((alError = alGetError()) != AL_NO_ERROR)
		throw std::runtime_error("alGenSources() failed");

	memset(testToneBuffer, 0, sizeof(testToneBuffer));

	TestTone();

	alBufferData(alBufferId, AL_FORMAT_MONO16, testToneBuffer, sizeof(testToneBuffer), 48000);
	if ((alError = alGetError()) != AL_NO_ERROR)
		throw std::runtime_error("alBufferData() failed");

	alSourcei(alSourceId, AL_BUFFER, alBufferId);
	if ((alError = alGetError()) != AL_NO_ERROR)
		throw std::runtime_error("alSource() failed");

	Play();
}

XAL::~XAL() {
	xprintf("XAL::~XAL()\n");
}

void XAL::Play() {
	alSourcePlay(alSourceId);
}

void XAL::Pause() {
	alSourcePause(alSourceId);
}

void XAL::Stop() {
	alSourceStop(alSourceId);
}

void XAL::TestTone() {
	int NSAMPLES = sizeof(testToneBuffer) / sizeof(testToneBuffer[0]);
	for (int i = 0; i < NSAMPLES; i++) {
		double value = 2.0 * (double)i / (double)128 * M_PI;
		testToneBuffer[i] = (short)(sin(value) * 32767.0);
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