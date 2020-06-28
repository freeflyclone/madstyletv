#include "ExampleXGL.h"

#include "XOuster.h"

XOuster::XOuster(std::string fn) : fileName(fn), XGLPointCloud(0)
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

	inputFileStream.open(fileName, std::ifstream::binary);

	if (inputFileStream) {
		SkipToNextFrame();

		// beginning of 1st full frame (messageId == 0)
		firstFrameOffset = inputFileStream.tellg();

		ReadSensorFrame();
	}
}

void XOuster::ReadSensorFrame()
{
	int frameSize = sizeof(ousterPacketBlock) * nPacketBlocks;
	int totalOffset = firstFrameOffset + frameIdx * frameSize;

	inputFileStream.seekg(totalOffset, std::ios_base::beg);

	for (int blockNum = 0; blockNum < nPacketBlocks; blockNum++)
	{
		inputFileStream.read((char*)&packetBlock, sizeof(packetBlock));
		if (!inputFileStream)
			throwXException("Failed to read complete UDP block");

		for (int i = 0; i < nAzimuthBlocks; i++) {
			ousterAzimuthBlock& ab = packetBlock[i];
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

void XOuster::SkipToNextFrame()
{
	if (inputFileStream)
	{
		ousterAzimuthBlock ab;

		inputFileStream.read((char*)&ab, sizeof(ab));
		if (!inputFileStream)
			throwXException("Failed to read ousterAzimuthBlock");

		// Skip to start of complete scan, if needed.
		if (ab.mId != 0) {
			do {
				inputFileStream.read((char*)&ab, sizeof(ab));
				if (!inputFileStream)
					throwXException("Failed to read AzimuthBlock");

				// if we detect max measurement Id, we're done.
				if (ab.mId == (nColumns - 1))
					break;
			} while (inputFileStream);
		}
	}
}

void XOuster::ReadConfig() {
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
	std::string modeFps = lidarMode.substr(lidarMode.find(delimiter) + 1);

	nColumns = std::stoi(modeColumns);
	nFps = std::stoi(modeFps);
	nPacketBlocks = nColumns / nAzimuthBlocks;

	int frameSize = sizeof(ousterPacketBlock) * nPacketBlocks;
	xprintf("nBlocks: %d, sizeof(ousterUdpBlock): %d, frameSize: %d\n", nPacketBlocks, sizeof(ousterPacketBlock), frameSize);
}

void XOuster::StepFrame(int delta)
{
	if (delta < 0)
	{
		if (frameIdx + delta < 0)
			frameIdx = 0;
		else
			frameIdx += delta;
	}
	else if (delta > 0)
	{
		if (frameIdx + delta > nMaxFrameNum)
			frameIdx = nMaxFrameNum;
		else
			frameIdx += delta;
	}

	if (delta)
	{
		v.clear();
		ReadSensorFrame();
		Load(shader, v);
	}
}

XOuster::~XOuster()
{
	inputFileStream.close();
}
