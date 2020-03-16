/*
** Note:
**
** This is a work in progress... I have obvious (to me) holes
** in my knowledge due to my "informal education" approach,
** and this object is at the core of my inter-thread communications.
***
** I want to prove to my satisfaction that it does what I think
** it does.  So this is a deep dive into this very particular feature,
** and there is a fair bit of availble info that has highlighted
** aspects of the problem that I was ignorant of, and now realize
** they deserve my full attention.
**
** In particular: I was unaware of so called "false sharing"
** in lockless designs, and now I get it.
**
** Bottom line: I need to analyze the performance for my use
** case(s), it may turn out that false sharing isn't a factor.
**
** 
*/

/**************************************************************
** XFifo
**
** Provides a thread safe circular byte buffer, optimized
** for inter-thread producer/consumer use, specifically
** in multimedia audio / video buffering with upper bounds
** on data rate, such that integer overflow of the 64 bit 
** read/write indexes is deterministically improbable before 
** the application exits. 
**
** IOW: pretend max read/write index value is infinity.
**
** Lockless design constraints: (to keep it simple)
**  Specific use case: single producer, single consumer each
**    in its own separate thread.
**  Power of two buffer size (buffer wraparound without division)
**  64bit write/read indexes (avoid integer overflow considerations)
**  Use memcpy() for Read() & Write() (for performance)
**  Read() / Write() return number of bytes read/written
**  Use std::atomic operations on read/write indexes.
**
** Semantics:
**  Both read and write functions have a timeout, in milliseconds,
**  of how long to wait for data available / room available,
**  respectively before returning how much was actually read / written
**  
**************************************************************/
#ifndef XFIFO_H
#define XFIFO_H

#include <stdint.h>
#include <atomic>
#include <stdexcept>
#include <limits>
#include <thread>
#include "xutils.h"

// This number matters, cross-platform requires runtime determination.
// Assume Intel x86 for now.
const int cacheLineSizeInBytes{ 64 };

template<typename T>
struct cacheLineStorage
{
	alignas(cacheLineSizeInBytes) T data;
	char pad[cacheLineSizeInBytes > sizeof(T) ? cacheLineSizeInBytes - sizeof(T) : 1];
};

class XFifo {
public:
	// max timeout (infinite wait) is 21 days (ish) is "virtually infinite"
	const int maxTimeout{ std::numeric_limits<int>::max() };

	XFifo(uint64_t s = 0x8000, int pi = 1) : size(s), pollingIntervalInMillis(pi) 
	{
		uint64_t leftmostBit = 0;
		int numBitsToTest = sizeof(size) * 8;

		for (int i = 0; i < numBitsToTest; i++)
			leftmostBit = ((size >> i) & 1) ? i : leftmostBit;

		if (size > (uint64_t(1) << leftmostBit))
			leftmostBit++;

		size = uint64_t(1) << leftmostBit;
		moduloMask = size - 1;

		buf = new uint8_t[size];

		if (!buf)
			throw std::runtime_error("Expected opertor new to throw, but it didn't. So here we are.");
	}

	~XFifo() 
	{
		delete buf;
	}

	uint64_t Write(uint8_t* data, uint64_t length, int timeout = 0) 
	{
		uint64_t available = Available();

		if (length > size)
			length = size;

		while (!abort && (available < length) && (timeout > 0))
		{
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(pollingIntervalInMillis));
			available = Available();
			timeout -= pollingIntervalInMillis;
		}

		if (0 == available)
			return 0;

		uint64_t actualWriteLength = (available < length) ? available : length;
		uint64_t nearnessToEnd = size - (wIdx.data & moduloMask);
		uint64_t firstSegmentSize = (actualWriteLength < nearnessToEnd) ? actualWriteLength : nearnessToEnd;
		uint64_t secondSegmentSize = actualWriteLength - firstSegmentSize;

		uint64_t actualBufferOffset = wIdx.data & moduloMask;

		memcpy(buf + actualBufferOffset, data, firstSegmentSize);
		wIdx.data += firstSegmentSize;

		if (secondSegmentSize)
		{
			auto actualBufferOffset = wIdx.data & moduloMask;
			memcpy(buf + actualBufferOffset, data, secondSegmentSize);
			wIdx.data += secondSegmentSize;
		}

		return actualWriteLength;
	}
	
	uint64_t Read(uint8_t* data, uint64_t length, int timeout = 0)
	{
		uint64_t used = Used();

		if (length > size)
			length = size;

		while (!abort && (used < length) && (timeout > 0))
		{
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(pollingIntervalInMillis));
			used = Used();
			timeout -= pollingIntervalInMillis;
		}

		if (0 == used)
			return 0;

		uint64_t actualReadLength = (used < length) ? used : length;
		uint64_t nearnessToEnd = size - (rIdx.data & moduloMask);
		uint64_t firstSegmentSize = (actualReadLength < nearnessToEnd) ? actualReadLength : nearnessToEnd;
		uint64_t secondSegmentSize = actualReadLength - firstSegmentSize;

		uint64_t actualBufferOffset = rIdx.data & moduloMask;

		memcpy(data, buf + actualBufferOffset, firstSegmentSize);
		rIdx.data += firstSegmentSize;

		if (secondSegmentSize)
		{
			auto actualBufferOffset = rIdx.data & moduloMask;
			memcpy(data, buf + actualBufferOffset, secondSegmentSize);
			rIdx.data += secondSegmentSize;
		}

		return actualReadLength;
	}

	uint64_t Available()
	{
		uint64_t w = wIdx.data;
		uint64_t r = rIdx.data;
		return size - (w - r);
	}

	uint64_t Used()
	{
		uint64_t w = wIdx.data;
		uint64_t r = rIdx.data;

		return w - r;
	}

	void AbortIO() 
	{
		abort = true;
	}

	void RestartIO()
	{
		abort = false;
	}

	void Reset()
	{
		memset(buf, 0, size);
		wIdx.data = rIdx.data = 0;
	}

	uint64_t Capacity()
	{
		return size;
	}

private:
	cacheLineStorage<std::atomic<uint64_t> > wIdx{ 0 };
	cacheLineStorage<std::atomic<uint64_t> > rIdx{ 0 };

	uint8_t* buf;

	uint64_t size;
	uint64_t moduloMask;

	int pollingIntervalInMillis;
	bool abort{ false };
};


#endif // XFIFO_H