/*
** File: XOuster.h
**
** Let's work with Ouster LIDAR data!
** Rudimentary file reading, just so I can get actual data
** to play with.
**
** Since sensors are UDP, a decent stream IO scheme is in order.
**
** Thankfully, FFFFFFFF precedes the first word of an
** azimuth block, and that looks to be unlikely in the data
** itself, so maybe it's safe to use for frame sync purposes.
**************************************************************/
#ifndef XOUSTER_H
#define XOUSTER_H

#include <locale>
#include <codecvt>
#include <iostream>
#include <fstream>
#include <cstdint>

// Default struct alignment is 8 in VS 2017 64-bit. We want 4
// This is little-endian CPU version.
#pragma pack(push, 4)
struct ousterAzimuthBlock
{
	uint64_t ts;
	uint16_t mId;
	uint16_t fId;
	uint32_t encCount;
	struct {
		uint32_t range;
		uint16_t signal;
		uint16_t reflect;
		uint32_t noise;
	} db[64];
	uint32_t azimuthBlockStatus;
};
typedef ousterAzimuthBlock ousterPacketBlock[16];
#pragma pack(pop)

class XOuster : public XGLPointCloud {
public:
	XOuster(std::string fn);
	~XOuster();

	void ReadSensorFrame();
	void SkipToNextFrame();
	void ReadConfig();
	void StepFrame(int delta);

private:
	std::string fileName;
	std::ifstream inputFileStream;
	int firstFrameOffset{ 0 };

	ousterPacketBlock packetBlock;

	int nColumns{ 1024 };
	static const int nRows{ 64 };
	static const int nAzimuthBlocks{ 16 };
	int nFps{ 10 };
	int nPacketBlocks{ 1024 / 16 };
	int nMaxFrameNum{ 2086 };
	int frameIdx{ 0 };

	std::vector<float> beamAltitudeAngles;
	std::vector<float> beamAzimuthAngles;
	std::string lidarMode;
};

#endif //ifndef XOUSTER