#include "xh264.h"

extern "C" {
	#include "configfile.h"
}

namespace {
	InputParameters inputParameters{ 0 };
}

Xh264Decoder::Xh264Decoder() : XThread("Xh264Decoder")
{
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
		int rc = DecodeOneFrame(&pDecPicList);
		if (rc == DEC_SUCCEED) {
			xprintf("New Frame: ");
			DecodedPicList *p = pDecPicList;
			do {
				if (p->iWidth)
					xprintf("Width: %d, Height: %d\n", p->iWidth, p->iHeight);
				p = p->pNext;
			} while (p != NULL);
		}
	}

	FinitDecoder(&pDecPicList);
	CloseDecoder();

	xprintf("Xh264Decoder::Run() finished\n");
}


