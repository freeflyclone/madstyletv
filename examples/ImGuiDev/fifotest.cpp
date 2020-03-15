#include "fifotest.h"

namespace XFifoTest
{
	Writer::Writer(XFifo* f) : pFifo(f), XThread("XFifoWriter")
	{
		for (int i = 0; i < poolWriteSize; i++)
			bp.push_back(BufferItem(poolWriteBufferSize, i));
		xprintf("%s()\n", __FUNCTION__);
	};

	void Writer::Run()
	{
		xprintf("%s() starting...\n", __FUNCTION__);
		int rotatingIndex = 0;

		while (IsRunning())
		{
			uint64_t writeChunkSize{ 0x300d }; // randomize this
			uint64_t totalWriteLength{ 0 };

			do {
				uint64_t nWritten = pFifo->Write(bp[rotatingIndex].data(), writeChunkSize - totalWriteLength);
				totalWriteLength += nWritten;

				if (totalWriteLength < writeChunkSize)
				{
					xprintf("%s() - waiting for room @ index: %d, need %llu bytes of room\n", __FUNCTION__, rotatingIndex, writeChunkSize - totalWriteLength);
					std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
				}
			} while (totalWriteLength < writeChunkSize);

			rotatingIndex = (rotatingIndex + 1) % poolWriteSize;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		}
		xprintf("%s() done.\n", __FUNCTION__);
	}

	Writer::~Writer()
	{
		if (IsRunning())
			Stop();

		bp.clear();
		xprintf("%s()\n", __FUNCTION__);
	}

	Reader::Reader(XFifo* f) : pFifo(f), XThread("XFifoReader")
	{
		bi.resize(poolReadBufferSize);

		for (int i = 0; i < poolReadSize; i++)
			bp.push_back(BufferItem(poolReadBufferSize, 0));
		xprintf("%s()\n", __FUNCTION__);
	};

	void Reader::Run()
	{
		xprintf("%s()\n", __FUNCTION__);
		while (IsRunning())
		{
			uint64_t nUsed = pFifo->Used();
			xprintf("%s() - Fifo level: %d\n", __FUNCTION__, nUsed);

			uint64_t nRead = pFifo->Read(bi.data(), bi.size());
			xprintf("%s() - Read() returned %d bytes\n", __FUNCTION__, nRead);

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(50));
		}
		xprintf("%s() done.\n", __FUNCTION__);
	}

	Reader::~Reader()
	{
		if (IsRunning())
			Stop();

		bp.clear();
		bi.clear();
		xprintf("%s()\n", __FUNCTION__);
	}

	Tester::Tester(XFifo *f) : pFifo(f)
	{
		reader = new Reader(pFifo);
		writer = new Writer(pFifo);
		xprintf("%s()\n", __FUNCTION__);
	}
	Tester::~Tester()
	{
		delete reader;
		delete writer;
		xprintf("%s()\n", __FUNCTION__);
	}
}