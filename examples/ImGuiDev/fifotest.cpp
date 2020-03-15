#include "fifotest.h"
#include "xlog.h"

XLOG_DECLARE("XFifoTest")

namespace XFifoTest
{
	Writer::Writer(XFifo* f) : pFifo(f), XThread("XFifoWriter")
	{
		for (int i = 0; i < poolWriteSize; i++)
			bp.push_back(BufferItem(poolWriteBufferSize, i));
		XLOG("%s()", __FUNCTION__);
	};

	void Writer::Run()
	{
		XLOG("%s() starting...", __FUNCTION__);
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
					XLOG("%s() - waiting for room @ index: %d, need %llu bytes of room", __FUNCTION__, rotatingIndex, writeChunkSize - totalWriteLength);
					std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
				}
			} while (totalWriteLength < writeChunkSize);

			rotatingIndex = (rotatingIndex + 1) % poolWriteSize;
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		}
		XLOG("%s() done.", __FUNCTION__);
	}

	Writer::~Writer()
	{
		if (IsRunning())
			Stop();

		bp.clear();
		XLOG("%s()", __FUNCTION__);
	}

	Reader::Reader(XFifo* f) : pFifo(f), XThread("XFifoReader")
	{
		bi.resize(poolReadBufferSize);

		for (int i = 0; i < poolReadSize; i++)
			bp.push_back(BufferItem(poolReadBufferSize, 0));
		XLOG("%s()", __FUNCTION__);
	};

	void Reader::Run()
	{
		XLOG("%s()", __FUNCTION__);
		while (IsRunning())
		{
			uint64_t nUsed = pFifo->Used();
			XLOG("%s() - Fifo level: %d", __FUNCTION__, nUsed);

			uint64_t nRead = pFifo->Read(bi.data(), bi.size());
			XLOG("%s() - Read() returned %d bytes", __FUNCTION__, nRead);

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(50));
		}
		XLOG("%s() done.", __FUNCTION__);
	}

	Reader::~Reader()
	{
		if (IsRunning())
			Stop();

		bp.clear();
		bi.clear();
		XLOG("%s()", __FUNCTION__);
	}

	Tester::Tester(XFifo *f) : pFifo(f)
	{
		reader = new Reader(pFifo);
		writer = new Writer(pFifo);
		XLOG("%s()", __FUNCTION__);
	}

	Tester::~Tester()
	{
		delete reader;
		delete writer;
		XLOG("%s()", __FUNCTION__);
	}
}