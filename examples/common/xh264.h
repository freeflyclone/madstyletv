#ifndef XH264_H
#define XH264_H

#include "xutils.h"
#include "xthread.h"

extern "C" {
#include "h264decoder.h"
};

class Xh264Decoder : public XThread
{
public:
	class Xh264AnnexB;

	Xh264Decoder();
	~Xh264Decoder();

	void Run();

private:
	Xh264AnnexB *pAnnexB;
};

#endif