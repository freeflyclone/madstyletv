#ifndef XH264_H
#define XH264_H

#include "xutils.h"
#include "xthread.h"

class Xh264Decoder : public XThread
{
public:
	Xh264Decoder();
	~Xh264Decoder();

	void Run();

private:
};

#endif