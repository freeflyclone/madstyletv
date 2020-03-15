#pragma once

#include "ExampleXGL.h"
#include "xglimgui.h"
#include "fifotest.h"

class XFifoTestControls : public XGLImGui
{
public:
	bool fifoTestWindow{ true };
	int writeBufferSize{ XFifoTest::poolWriteBufferSize };
	int readBufferSize{ XFifoTest::poolReadBufferSize };
	int writePoolSize{ XFifoTest::poolWriteSize };
	int readPoolSize{ XFifoTest::poolReadSize };

	bool isRunning{ false };
	bool isReading{ true };
	bool isWriting{ true };
};
