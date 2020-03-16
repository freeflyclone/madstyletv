#include "fifotest.h"
#include "xlog.h"

XLOG_DECLARE("XFifoTest");

namespace XFifoTest
{
	Writer::Writer(XFifo* f) : pFifo(f), XThread("XFifoWriter")
	{
		for (int i = 0; i < poolWriteSize; i++)
			bp.push_back(BufferItem(poolWriteBufferSize, i));
		XLOG();
	};

	void Writer::Run()
	{
		XLOG("starting...");
		int rotatingIndex = 0;

		while (IsRunning())
		{
			uint64_t writeChunkSize{ 0x300d }; // randomize this
			uint64_t totalWriteLength{ 0 };

			do {
				totalWriteLength += pFifo->Write(bp[rotatingIndex].data(), writeChunkSize - totalWriteLength, 10);
			} while (totalWriteLength < writeChunkSize && IsRunning());

			rotatingIndex = (rotatingIndex + 1) % poolWriteSize;
		}
		XLOG("done.");
	}

	Writer::~Writer()
	{
		if (IsRunning()) 
		{
			pFifo->AbortIO();
			WaitForStop();
		}

		bp.clear();
		XLOG("");
	}

	Reader::Reader(XFifo* f) : pFifo(f), XThread("XFifoReader")
	{
		bi.resize(poolReadBufferSize);

		for (int i = 0; i < poolReadSize; i++)
			bp.push_back(BufferItem(poolReadBufferSize, 0));
		XLOG("");
	};

	void Reader::Run()
	{
		XLOG("starting...");
		FILE *of = fopen("fifoTestReader.bin", "wb");

		while (IsRunning())
		{
			uint64_t nRead = pFifo->Read(bi.data(), bi.size(), 10);

			if (nRead)
				fwrite(bi.data(), 1, nRead, of);
			else
				std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		}

		fclose(of);
		XLOG("done.");
	}

	Reader::~Reader()
	{
		if (IsRunning())
		{
			pFifo->AbortIO();
			WaitForStop();
		}

		bp.clear();
		bi.clear();
		XLOG("");
	}

	Tester::Tester(XFifo *f) : pFifo(f)
	{
		reader = new Reader(pFifo);
		writer = new Writer(pFifo);
		XLOG("");
	}

	Tester::~Tester()
	{
		delete reader;
		delete writer;
		XLOG("");
	}

	bool Tester::IsRunning()
	{
		return reader->IsRunning() || writer->IsRunning();
	}

	void Tester::Start()
	{
		StartReader();
		StartWriter();
	}

	void Tester::Stop()
	{
		StopWriter();
		StopReader();
	}

	void Tester::StartReader()
	{
		reader->Start();
	}
	void Tester::StopReader()
	{
		reader->WaitForStop();
	}
	void Tester::StartWriter()
	{
		writer->Start();
	}
	void Tester::StopWriter()
	{
		writer->WaitForStop();
	}
}