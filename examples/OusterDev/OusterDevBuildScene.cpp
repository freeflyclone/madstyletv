/**************************************************************
** OusterDevBuildScene.cpp
**
** Let's experiment with Ouster Lidar Sensor data!
**************************************************************/
#include "ExampleXGL.h"

// Default struct alignment is 8 in VS 2017 64-bit. We want 4
#pragma pack(push, 4)
struct ousterAzimuthBlock
{
	uint64_t ts;
	uint16_t fId;
	uint16_t mId;
	uint32_t encCount;

	struct {
		uint32_t range;
		uint16_t signal;
		uint16_t reflect;
		uint32_t noise;
	} db[64];
	uint32_t azimuthBlockStatus;
};
#pragma pack(pop)

class OusterSensor {
public:
	OusterSensor(std::string fn) : fileName(fn)
	{
		xprintf("%s: fileName: %s, sizeof(ousterAzimuthBlock): %d\n", 
			__FUNCTION__, 
			fileName.c_str(), 
			sizeof(ousterAzimuthBlock));

		if ((fp = fopen(fileName.c_str(), "rb")) == nullptr)
			throw std::runtime_error("Unable to open file.");

		// flush turd at start of file
		for (int i = 0; i < 0xE0; i++)
			fread(&ab, sizeof(ab), 1, fp);

		// read one scan's worth of data
		for (int i = 0; i < nColumns; i++) {
			fread(&ab, sizeof(ab), 1, fp);

			for (int y = 0; y < nRows; y++) {
				uint32_t* p = rBuffer + i + (y * nColumns);
				*p = ab.db[y].range;
			}

			xprintf("%016X, %04X %04X, %d - %08X, %08X, %04X, %04X, %08X\n", 
				ab.ts, 
				ab.mId, 
				ab.fId,
				ab.encCount,
				ab.azimuthBlockStatus,
				ab.db[0].range,
				ab.db[0].signal,
				ab.db[0].reflect,
				ab.db[0].noise);
		}
	}

	~OusterSensor() 
	{
	}

private:
	std::string fileName;
	FILE *fp;

	ousterAzimuthBlock ab;

	static const int nRows{ 64 };
	static const int nColumns{ 1024 };
	uint32_t rBuffer[nRows*nColumns];
};

void ExampleXGL::BuildScene() {
	try
	{
		OusterSensor os("C:/Users/evan/Desktop/lombard_street_OS1.raw");
	}
	catch (std::exception e)
	{
		xprintf("OusterSensor error: %s\n", e.what());
	}
}
