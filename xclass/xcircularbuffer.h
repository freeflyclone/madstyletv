/**************************************************************
** XCircularBuffer
**
** Provides a thread safe byte oriented circular buffer for
** inter-module communication in multimedia applications. 
**
** Derived from old C code, not yet fully C++ified
**************************************************************/
#ifndef XCIRCULARBUFFER_H
#define XCIRCULARBUFFER_H

#include <stdexcept>
#include <mutex>

#include <xthread.h>

namespace {
	int SizeAdjust(unsigned int s)	{
		unsigned int i,size;

		// Ensure that "size" gets rounded up to nearest power of 2, so...
		// ...get to the first bit of "size" that is set...
		for (i = 1 << (8 * sizeof(s) - 1); (s&i) == 0; i >>= 1);
		// ... and remember it.
		size = i;
		// Scan remaing bits, if one is set, look no further.
		for (i >>= 1; i; i >>= 1)
			if (s&i)
				break;
		// if bailed out of above loop, gotta round up
		if (i)
			size <<= 1;

		// size is now next larger power-of-two
		return size;
	}
};

class XCircularBuffer
{
public:
	XCircularBuffer(unsigned int s) {
		size = SizeAdjust(s);
		sizeMask = size - 1;
		rIdx = wIdx = 0;

		moreRoom.notify();

		if ((buff = new unsigned char[size]) == NULL)
			throw std::runtime_error("Unable to allocate circular buffer");
	}

	~XCircularBuffer() {
		if (buff)
			delete buff;
	}
	void Write(unsigned char c) {
		std::lock_guard<std::mutex> lock(mutex);

		// store in rxBuff[], circular style
		buff[wIdx & sizeMask] = c;
		wIdx++;

		// adjust read index if buffer is full
		// (blindly discarding stale data, client (ReadByte() caller) better keep up)
		if ((wIdx - rIdx) > size)
			rIdx = wIdx - size;

		notEmpty.notify();
	}

	int Write(unsigned char *s, unsigned int count) {
		int emptySpace;
		// wait for sufficient room for "count".
		//for (int emptySpace = Size() - Count(); emptySpace < count; moreRoom.wait());
		do {
			emptySpace = Size() - Count();
			bool waitTimeOut = moreRoom.wait_for(500);
			if (!waitTimeOut)
				break;
		} while (emptySpace < count);

		if (emptySpace < count)
			return 0;

		std::lock_guard<std::mutex> lock(mutex);

		for (unsigned int i = 0; i<count; i++) {
			buff[wIdx & sizeMask] = s[i];
			wIdx++;
		}

		// adjust read index if buffer is full
		// (blindly discarding stale data, client (ReadByte() caller) better keep up)
		if ((wIdx - rIdx) > size)
			rIdx = wIdx - size;

		notEmpty.notify();
		return count;
	}

	bool Read(unsigned char *c) {
		if (rIdx == wIdx)
			return false;

		notEmpty.wait();

		std::lock_guard<std::mutex> lock(mutex);

		// read from buffer, circular style
		*c = buff[rIdx & sizeMask];
		rIdx++;

		moreRoom.notify();
		return true;
	}

	int Read(unsigned char *s, unsigned int size) {
		int count = 0;

		// insert timed wait for data to be available here
		notEmpty.wait();

		std::lock_guard<std::mutex> lock(mutex);

		if (rIdx == wIdx || !buff)
			return false;

		for (unsigned int i = 0; (i<size) && (rIdx < wIdx); i++) {
			s[i] = buff[rIdx & sizeMask];
			rIdx++;
			count++;
		}

		moreRoom.notify();
		return count;
	}

	int Count() {
		return (int)(wIdx - rIdx);
	}

	int Size() {
		return size;
	}

	void Flush() {
		if (buff) {
			wIdx = rIdx = 0;
		}
	}

	void Rewind(int howFar) {
		if (howFar < 0)
			return;

		rIdx -= howFar;
		if (rIdx < 0)
			rIdx = 0;

		moreRoom.notify();
	}

private:
	std::mutex		mutex;
	XSemaphore		notEmpty;
	XSemaphore		moreRoom;
	unsigned char	*buff;			// MUST point to buffer that's a power of two for easy circular buffer management
	unsigned int	size;			// MUST be initializedto size of buff
	int				sizeMask;		// MUST be initialized to size - 1;
	__int64			rIdx;			// it is HIGHLY unlikely that these will ever wrap around
	__int64			wIdx;			// ...even at high data rates
};

#endif