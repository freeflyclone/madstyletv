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
	xprintf("XAL::~XAL()\n");
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

void XAL::TestTone() {
	int NSAMPLES = sizeof(testToneBuffer) / sizeof(testToneBuffer[0]);

	memset(testToneBuffer, 0, sizeof(testToneBuffer));

	for (int i = 0; i < NSAMPLES; i++) {
		double value = 2.0 * (double)i / (double)128 * M_PI;
		testToneBuffer[i].left = (short)(sin(value) * 32767.0);
		testToneBuffer[i].right = (short)(sin(value*3) * 32767.0);
	}

	for (int i = 0; i < nBuffers; i++) {
		alBufferData(alBufferIds[i], format, testToneBuffer, sizeof(testToneBuffer), sampleRate);
		AL_CHECK("alBufferData() failed");
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