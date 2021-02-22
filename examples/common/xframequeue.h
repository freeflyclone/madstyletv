#ifndef XFRAMEQUEUE_H
#define XFRAMEQUEUE_H

#include <queue>
#include <atomic>
#include <thread>
#include "xclasses.h"

class XFrameQueue {
public:
	typedef uint16_t* Frame;
	typedef std::queue<Frame> Queue;

	XFrameQueue();
	~XFrameQueue();

	void Post(Frame fn);
	Frame& Remove();

private:
	std::mutex m_lock;
	Queue m_queue;
	XSemaphore m_signal;
	Frame m_pFrame;
	Frame m_nullFrame = (Frame)(nullptr);
};



#endif
