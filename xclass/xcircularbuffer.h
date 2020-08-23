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
	XCircularBuffer(size_t s = 0x8000) : size(s), rIdx(0), wIdx(0) {
		if ((buff = new uint8_t[size]) == NULL)
			throw std::runtime_error("XCircularBuffer::XCircularBuffer(): Unable to allocate circular buffer");
	}

	~XCircularBuffer() {
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
		while (Count() < n)
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));

		std::lock_guard<std::mutex> lock(mutexLock);
		for (size_t i = 0; i<n; i++) {
			*b++ = buff[rIdx&(size - 1)];
			rIdx++;
		}

		return n;
	}

	uint64_t Skip(uint64_t n) { 
		//while (Count() < n)
			//std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));

		std::lock_guard<std::mutex> lock(mutexLock);
		if (rIdx + n < wIdx) {
			rIdx += n;
			n = 0;
		}
		else {
			n -= wIdx - rIdx;
			rIdx = wIdx;
		}
		return n;
	}

	uint64_t Count() { 
		std::lock_guard<std::mutex> lock(mutexLock);
		return wIdx - rIdx;
	}
	size_t Size() { return size; }

//private:
	size_t size;
	uint8_t *buff;
	uint64_t rIdx;
	uint64_t wIdx;
	std::mutex mutexLock;
};


#endif
