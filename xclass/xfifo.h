/**************************************************************
** XFifo
**
** Provides a thread safe object oriented circular buffer for
** inter-module communication in multimedia applications.
**
** Derived from old C code, not yet fully C++ified, and not
** particularly efficient either.
**
** Size must be power of 2.  This is not enforced, and data
** corruption WILL occur if it is not obeyed.
**************************************************************/
#ifndef XFIFO_H
#define XFIFO_H

#include <stdexcept>
#include <mutex>
#include <queue>

#include <xthread.h>

template <class T>
class XFifo {
public:
	XFifo(int s) : emptyCount(s) {
	}

	~XFifo() {
	}

	void Put(T elem) {
		emptyCount.wait_for(100);
		elems.push(elem);
		fullCount.notify();
	}

	T Get() {
		fullCount.wait_for(100);
		T elem = elems.front();
		elems.pop();
		emptyCount.notify();

		return elem;
	}

	size_t Size() {
		return elems.size();
	}

private:
	std::queue<T> elems;
	XSemaphore emptyCount;
	XSemaphore fullCount;
};


#endif // XFIFO_H