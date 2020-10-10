#include "xdispatchq.h"

XDispatchQueue::XDispatchQueue() : m_thread(&XDispatchQueue::Run, this) {
	m_begin = true;
};

XDispatchQueue::XDispatchQueue(const char* queueName) : m_name(queueName) {
	FUNC("name: %s\n", m_name.c_str());
}

XDispatchQueue::~XDispatchQueue() {
	m_isRunning = false;
	m_thread.join();
}

void XDispatchQueue::Run() {
	while (!m_begin)
		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1));

	while (m_isRunning) {
		// Using wait_for() eases thread termination, by setting m_isRunning to false
		if (m_signal.wait_for(250)) {
			auto fn = Remove();
			fn();
		}
	}
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

