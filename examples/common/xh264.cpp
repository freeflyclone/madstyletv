#include "xh264.h"

extern "C" {
	#include "configfile.h"
}

namespace {
	InputParameters inputParameters{ 0 };
}

Xh264Decoder::Xh264Decoder() : XThread("Xh264Decoder")
{
	InputParameters& ip = inputParameters;

	init_time();

	// this also parses the default "Decoder.cfg" file
	ParseCommand(&inputParameters, 0, nullptr);
}

Xh264Decoder::~Xh264Decoder()
{

}

void Xh264Decoder::Run() {
	DecodedPicList *pDecPicList;

	OpenDecoder(&inputParameters);

	while (IsRunning())
	{
		DecodeOneFrame(&pDecPicList);
	}

	xprintf("Xh264Decoder::Run() finished\n");
}


