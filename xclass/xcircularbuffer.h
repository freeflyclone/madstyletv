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

#include <xthread.h>

class XCircularBuffer {
public:
	XCircularBuffer(int s) : size(s), rIdx(0), wIdx(0), emptyCount(s) {
		if ((buff = new unsigned char[size]) == NULL)
			throw std::runtime_error("Circular::Circular(): Unable to allocate circular buffer");
	}

	~XCircularBuffer() {
		if (buff)
			delete buff;
	}

	int Write(unsigned char *b, unsigned int n) {
		while (n) {
			emptyCount.wait_for(100);
			buff[wIdx&(size - 1)] = *b++;
			wIdx++;
			n--;
			fullCount.notify();
		}
		return n;
	}

	int Read(unsigned char *b, unsigned int n) {
		while (n) {
			fullCount.wait_for(100);
			*b++ = buff[rIdx&(size - 1)];
			rIdx++;
			n--;
			emptyCount.notify();
		}
		return n;
	}

	int Count() {
		return int(wIdx - rIdx);
	}

private:
	const int size;
	unsigned char *buff;
	std::atomic<std::uint64_t> rIdx;
	std::atomic<std::uint64_t> wIdx;
	XSemaphore emptyCount;
	XSemaphore fullCount;
};


#endif
