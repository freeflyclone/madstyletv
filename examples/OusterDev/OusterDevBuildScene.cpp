/**************************************************************
** OusterDevBuildScene.cpp
**
** Let's experiment with Ouster Lidar Sensor data!
**************************************************************/
#include "ExampleXGL.h"

// Default struct alignment is 8 in VS 2017 64-bit. We want 4
#pragma pack(push, 4)

struct _data {
	uint32_t range;
	uint16_t signal;
	uint16_t reflect;
	uint32_t noise;
};

typedef _data dataBlock[64];

struct ousterAzimuthBlock
{
	uint64_t ts;
	uint16_t fId;
	uint16_t mId;
	uint32_t encCount;
	dataBlock db;
	uint32_t azimuthBlockStatus;
};
#pragma pack(pop)

class OusterSensor : public XGLPointCloud {
public:
	OusterSensor(std::string fn) : fileName(fn), XGLPointCloud(64 * 1024)
	{
		ReadConfig();

		xprintf("%s: fileName: %s, sizeof(ousterAzimuthBlock): %d\n",
			__FUNCTION__,
			fileName.c_str(),
			sizeof(ousterAzimuthBlock));

		if ((fp = fopen(fileName.c_str(), "rb")) == nullptr)
			throw std::runtime_error("Unable to open file.");

		// flush turd at start of file
		for (int i = 0; i < 0xE0; i++)
			fread(&ab, sizeof(ab), 1, fp);

		v.clear();

		xprintf("sizeof(dataBlock): %d\n", sizeof(dataBlock));

		// read one scan's worth of data
		for (int i = 0; i < nColumns; i++) {
			fread(&ab, sizeof(ab), 1, fp);

			xprintf("%016X, %04X %04X, %5d - %08X, %08X, %04X, %04X, %08X\n",
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

	void ReadConfig() {
		XAssets sensorCfg("C:/Users/evan/Desktop/lombard_street_config-1024x10.json");

		JSONArray beam_altitude_angles = sensorCfg.Find(L"beam_altitude_angles")->AsArray();
		JSONArray beam_azimuth_angles = sensorCfg.Find(L"beam_azimuth_angles")->AsArray();

		for (JSONValue* altitude : beam_altitude_angles)
			beamAltitudeAngles.push_back(altitude->AsNumber());

		for (JSONValue* azimuth : beam_azimuth_angles)
			beamAzimuthAngles.push_back(azimuth->AsNumber());
	}

	~OusterSensor()
	{
	}


private:
	std::string fileName;
	FILE *fp;

	ousterAzimuthBlock ab;

	static const int nColumns{ 1024 };
	static const int nRows{ 64 };

	uint32_t rBuffer[nColumns * nRows];

	std::vector<float> beamAltitudeAngles;
	std::vector<float> beamAzimuthAngles;

	XGLTexQuad* rangeImage;
};

void ExampleXGL::BuildScene() {
	OusterSensor *pOS;

	try
	{
		AddShape("shaders/000-simple", [&]() {
			pOS = new OusterSensor("C:/Users/evan/Desktop/lombard_street_OS1.raw");
			return pOS;
		});
	}
	catch (std::exception e)
	{
		xprintf("OusterSensor error: %s\n", e.what());
	}
}

/*
*/
