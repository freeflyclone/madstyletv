#include "fifotest.h"

namespace XFifoTest
{
	Writer::Writer(XFifo* f) : pFifo(f), XThread("XFifoWriter")
	{
		xprintf("%s(): pFifo: 0x%p\n", __FUNCTION__, pFifo);
	};

	void Writer::Run()
	{
	}

	Reader::Reader(XFifo* f) : pFifo(f), XThread("XFifoReader")
	{
		xprintf("%s(): pFifo: 0x%p\n", __FUNCTION__, pFifo);
	};

	void Reader::Run()
	{

	}
}