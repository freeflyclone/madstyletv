#pragma once

#include "xfifo.h"
#include "xthread.h"
#include "xutils.h"

namespace XFifoTest
{
	typedef char BufferItem[0x10000];
	typedef std::vector<BufferItem> BufferPool;

	class Writer : public XThread
	{
	public:
		Writer(XFifo *f);
		void Run();

	private:
		XFifo *pFifo{ nullptr };
	};

	class Reader : public XThread
	{
	public:
		Reader(XFifo *f);
		void Run();

	private:
		XFifo *pFifo{ nullptr };
	};

}