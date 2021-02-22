#include "xframequeue.h"

XFrameQueue::XFrameQueue() {

}

XFrameQueue::~XFrameQueue() {

}

void XFrameQueue::Post(Frame fn) {
	std::unique_lock<std::mutex> lock(m_lock);

	m_queue.push(fn);
	lock.unlock();
	m_signal.notify();
}

XFrameQueue::Frame& XFrameQueue::Remove() {
	std::unique_lock<std::mutex> lock(m_lock);

	if (m_queue.size()) {
		m_pFrame = m_queue.front();
		m_queue.pop();
		lock.unlock();
		return m_pFrame;
	}
	return m_nullFrame;
}
