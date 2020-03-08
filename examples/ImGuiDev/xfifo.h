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
#include "xutils.h"

class XFifo {
public:
	XFifo(uint64_t s, int pi = 10) : size(s), pollingIntervalInMillis(pi) 
	{
		uint64_t leftmostBit = 0;
		int numBitsToTest = sizeof(size) * 8;

		for (int i = 0; i < numBitsToTest; i++)
			leftmostBit = ((size >> i) & 1) ? i : leftmostBit;

		if (size > (uint64_t(1) << leftmostBit))
			leftmostBit++;

		size = uint64_t(1) << leftmostBit;

		// "new" will through if it must, so we don't need to
		buf = new uint8_t[size];
	}

	~XFifo() 
	{
		delete buf;
	}

	int Write(uint8_t* data, uint64_t length, int timeout) 
	{
	}
	
	int Read(uint8_t* data, uint64_t length, int timeout)
	{
	}

	uint64_t SpaceAvailable()
	{
		return size - (wIdx - rIdx);
	}

	uint64_t DataAvailable()
	{
		return (wIdx - rIdx);
	}

private:
	uint8_t* buf;

	uint64_t size;
	std::atomic<uint64_t> wIdx{ 0 };
	std::atomic<uint64_t> rIdx{ 0 };

	int pollingIntervalInMillis{ 1 };
};


#endif // XFIFO_H