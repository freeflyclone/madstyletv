/**************************************************************
** XCircularBuffer
**
** Provides a thread safe byte oriented circular buffer for
** inter-module communication in multimedia applications. 
**
** Derived from old C code, not yet fully C++ified, and not
** particularly efficient either.
**
** Size must be power of 2.  This is not enforced, and data
** corruption WILL occur if it is not obeyed.
**************************************************************/
#ifndef XCIRCULARBUFFER_H
#define XCIRCULARBUFFER_H

#include <stdexcept>
#include <mutex>
#include <atomic>
#include <cstdint>

#include <xthread.h>

class XCircularBuffer {
public:
	XCircularBuffer(size_t s) : size(s), rIdx(0), wIdx(0), emptyCount((int)s) {
		if ((buff = new uint8_t[size]) == NULL)
			throw std::runtime_error("Circular::Circular(): Unable to allocate circular buffer");
	}

	~XCircularBuffer() {
		if (buff)
			delete buff;
	}

	size_t Write(uint8_t *b, size_t n) {
		std::lock_guard<std::mutex> lock(mutexLock);

		for(size_t i=0; i<n; i++) {
			buff[wIdx&(size - 1)] = *b++;
			wIdx++;
		}

		return n;
	}

	size_t Read(uint8_t *b, size_t n) {
		std::lock_guard<std::mutex> lock(mutexLock);
		if (Count() < n)
			n = Count();
		for (int i = 0; i<n; i++) {
			*b++ = buff[rIdx&(size - 1)];
			rIdx++;
		}
		return n;
	}

	uint64_t Count() {
		return int(wIdx - rIdx);
	}

private:
	size_t size;
	uint8_t *buff;
	std::atomic<std::uint64_t> rIdx;
	std::atomic<std::uint64_t> wIdx;
	XSemaphore emptyCount;
	XSemaphore fullCount;
	std::mutex mutexLock;
};


#endif
