#pragma once

#include "xfifo.h"
#include "xthread.h"
#include "xutils.h"
#include "xgl.h"

namespace XFifoTest
{
	const int poolWriteSize = 32;
	const int poolReadSize = 16;
	const int poolWriteBufferSize = 0x10000;
	const int poolReadBufferSize = 0x1000;

	typedef std::vector<uint8_t>BufferItem;
	typedef std::vector<BufferItem> BufferPool;

	class Writer : public XThread
	{
	public:
		Writer(XFifo *f);
		~Writer();

		void Run();

	private:
		XFifo *pFifo{ nullptr };
		BufferPool bp;
	};

	class Reader : public XThread
	{
	public:
		Reader(XFifo *f);
		~Reader();

		void Run();

	private:
		XFifo *pFifo{ nullptr };
		BufferItem bi;
		BufferPool bp;
	};

	class Tester : public XGLShape 
	{
	public:
		Tester(XFifo* f);
		~Tester();

	private:
		Reader* reader;
		Writer* writer;
		XFifo *pFifo{ nullptr };
	};
}