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
#include "xutils.h"


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

	uint64_t Write(uint8_t* data, uint64_t length, int timeout) 
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

		auto actualWriteLength = (available < length) ? available : length;
		auto nearnessToEnd = size - wIdx;
		auto firstSegmentSize = (actualWriteLength < nearnessToEnd) ? actualWriteLength : nearnessToEnd;
		auto secondSegmentSize = actualWriteLength - firstSegmentSize;

		auto actualBufferOffset = wIdx & moduloMask;

		memcpy(buf + actualBufferOffset, data, firstSegmentSize);
		wIdx += firstSegmentSize;

		if (secondSegmentSize)
		{
			auto actualBufferOffset = wIdx & moduloMask;
			memcpy(buf + actualBufferOffset, data, secondSegmentSize);
			wIdx += secondSegmentSize;
		}

		return actualWriteLength;
	}
	
	uint64_t Read(uint8_t* data, uint64_t length, int timeout)
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

		auto actualReadLength = (used < length) ? used : length;
		auto nearnessToEnd = size - rIdx;
		auto firstSegmentSize = (actualReadLength < nearnessToEnd) ? actualReadLength : nearnessToEnd;
		auto secondSegmentSize = actualReadLength - firstSegmentSize;

		auto actualBufferOffset = rIdx & moduloMask;

		memcpy(data, buf + actualBufferOffset, firstSegmentSize);
		rIdx += firstSegmentSize;

		if (secondSegmentSize)
		{
			auto actualBufferOffset = rIdx & moduloMask;
			memcpy(data, buf + actualBufferOffset, secondSegmentSize);
			rIdx += secondSegmentSize;
		}

		return actualReadLength;
	}

	uint64_t Available()
	{
		uint64_t w = wIdx;
		uint64_t r = rIdx;
		return size - (w - r);
	}

	uint64_t Used()
	{
		uint64_t w = wIdx;
		uint64_t r = rIdx;

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
		wIdx = rIdx = 0;
	}

private:
	uint8_t* buf;

	uint64_t size;
	uint64_t moduloMask;
	std::atomic<uint64_t> wIdx{ 0 };
	std::atomic<uint64_t> rIdx{ 0 };

	int pollingIntervalInMillis;
	const int maxPossibleTimeout = std::numeric_limits<int>::max();
	bool abort{ false };
};


#endif // XFIFO_H