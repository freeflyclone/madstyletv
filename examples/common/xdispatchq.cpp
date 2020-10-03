#include "xdispatchq.h"

XDispatchQueue::XDispatchQueue() : XThread("DispatchQueue") {
};

XDispatchQueue::~XDispatchQueue() {
	Stop();
}

void XDispatchQueue::Post(Function fn) {
	std::unique_lock<std::mutex> lock(m_lock);

	m_queue.push(fn);
	lock.unlock();
	m_signal.notify();
};

XDispatchQueue::Function& XDispatchQueue::Remove() {
	std::unique_lock<std::mutex> lock(m_lock);

	if (m_queue.size()) {
		m_fn = m_queue.front();
		m_queue.pop();
		lock.unlock();
		return m_fn;
	}
	else
		return m_emptyFn;
}

void XDispatchQueue::Run() {
	while (IsRunning()) {
		// Using wait_for() allows periodic checking of IsRunning(),
		// which is useful for allowing XThread::Stop() to actually work.
		if (m_signal.wait_for(250)) {
			auto fn = Remove();
			fn();
		}
	}
}
