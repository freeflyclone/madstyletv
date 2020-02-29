#ifndef XH264_H
#define XH264_H

#include "xutils.h"
#include "xthread.h"
#include "XGL.h"

extern "C" {
#include "global.h"
#include "mbuffer.h"
}

class Xh264Decoder : public XThread, public XGLTexQuad
{
public:
	typedef std::function<void(VideoParameters*, StorablePicture*, int)> Callback;
	typedef std::vector<Callback> CallbackList;

	Xh264Decoder();
	~Xh264Decoder();

	static void _callback(VideoParameters*, StorablePicture*, int);

	void AddCallback(Callback);
	void InvokeCallbacks(VideoParameters*, StorablePicture*, int);

	void Draw();
	void Run();

	uint8_t yuvBuffer[1920 * 1080 * 3];
	CallbackList cbList;
};

#endif