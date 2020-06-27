/**************************************************************
** OusterDevBuildScene.cpp
**
** Let's experiment with Ouster Lidar Sensor data!
**************************************************************/
#include "ExampleXGL.h"

#include <locale>
#include <codecvt>
#include <iostream>
#include <fstream>

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
	uint16_t mId;
	uint16_t fId;
	uint32_t encCount;
	dataBlock db;
	uint32_t azimuthBlockStatus;
};

typedef ousterAzimuthBlock ousterUdpBlock[16];

#pragma pack(pop)

class OusterSensor : public XGLPointCloud {
public:
	OusterSensor(std::string fn) : fileName(fn), XGLPointCloud(0)
	{
		ReadConfig();

		XGLPointCloud::drawFn = [&]()
		{
			if (v.size())
			{
				glPointSize(2.0f);
				glDrawArrays(GL_POINTS, 0, GLsizei(v.size()));
				GL_CHECK("glDrawPoints() failed");
			}
		};

		ifs.open(fileName, std::ifstream::binary);
		if (ifs)
		{
			ousterAzimuthBlock ab;

			ifs.read((char*)&ab, sizeof(ab));
			if (!ifs)
				throw std::runtime_error("Failed to read initial AzimuthBlock");

			// Skip to start of complete scan, if needed.
			if (ab.mId != 0) {
				do {
					ifs.read((char*)&ab, sizeof(ab));
					if (!ifs)
						throw std::runtime_error("Failed to read AzimuthBlock");

					// if we detect (hard-coded for now) max measurement Id, we're done.
					if (ab.mId == (nColumns-1))
						break;
				} while (ifs);
			}
		}
		else 
			throw std::runtime_error("Failed to open input file");

		firstFrameOffset = ifs.tellg();

		for (int blockNum = 0; blockNum < nBlocks; blockNum++)
		{
			ifs.read((char*)&oub, sizeof(oub));
			if (!ifs)
				throw std::runtime_error("Failed to read complete UDP block");

			for (int i = 0; i < nAzimuthBlocks; i++) {
				ousterAzimuthBlock& ab = oub[i];
				float theta = (float)(blockNum * nAzimuthBlocks + i) / nColumns * M_PI * 2;
				auto x = sin(theta);
				auto y = cos(theta);

				for (int j = 0; j < nRows; j++) {
					float phi = beamAltitudeAngles[j] / 360.0 * M_PI * 2;
					float range = (float)ab.db[j].range / 1000.0f;

					auto xr = range * x;
					auto yr = range * y;
					auto z = range * sin(phi);

					v.push_back({ {xr,yr,z}, {}, {}, XGLColors::white });
				}
			}
		}
	}

	void ReadConfig() {
		XAssets sensorCfg("C:/Users/evan/Desktop/lombard_street_config-1024x10.json");

		JSONArray beam_altitude_angles = sensorCfg.Find(L"beam_altitude_angles")->AsArray();
		JSONArray beam_azimuth_angles = sensorCfg.Find(L"beam_azimuth_angles")->AsArray();

		std::wstring lidar_mode = sensorCfg.Find(L"lidar_mode")->AsString();
		// Pesky JSON uses 16 bit char, so do the codecvt thing to get std::string
		lidarMode = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(lidar_mode);

		for (JSONValue* altitude : beam_altitude_angles)
			beamAltitudeAngles.push_back(altitude->AsNumber());

		for (JSONValue* azimuth : beam_azimuth_angles)
			beamAzimuthAngles.push_back(azimuth->AsNumber());

		std::string delimiter("x");
		std::string modeColumns = lidarMode.substr(0, lidarMode.find(delimiter));
		std::string modeFps = lidarMode.substr(lidarMode.find(delimiter)+1);

		nColumns = std::stoi(modeColumns);
		nFps = std::stoi(modeFps);
		nBlocks = nColumns / nAzimuthBlocks;
	}

	~OusterSensor()
	{
	}


private:
	std::string fileName;
	std::ifstream ifs;
	int firstFrameOffset{ 0 };

	ousterUdpBlock oub;

	int nColumns{ 1024 };
	static const int nRows{ 64 };
	static const int nAzimuthBlocks{ 16 };
	int nFps{ 10 };
	int nBlocks{ 1024 / 16 };

	std::vector<float> beamAltitudeAngles;
	std::vector<float> beamAzimuthAngles;
	std::string lidarMode;
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
