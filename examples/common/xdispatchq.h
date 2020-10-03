#ifndef XDISPATCHQ_H
#define XDISPATCHQ_H

#include "xclasses.h"
#include <queue>
/**
* Simple queue to execute async Function objects on alternate thread(s)
* Intended for executing lambda functions, where "capturing" of in-scope
* variables is possible.
*
* Post() adds a Function to the back of the Queue
* Remove() extracts a Function from the front of the Queue
*
* XThread::Run() executes a timed wait_for() on an XSemaphore(), and if "true" is
* returned it then Remove()'s a Function and runs it in the thread.
* One must call XDispatchQueue::Start() to start the thread for background dequeueing & call of Function.
*
* Use of Remove() in multiple threads simultaneously is presently undefined.
*/
class XDispatchQueue : public XThread {
public:
	typedef std::function<void()> Function;
	typedef std::queue<Function> Queue;

	XDispatchQueue();
	~XDispatchQueue();

	void Post(Function fn);
	Function& Remove();
	
	// Required by XThread, which defines Run() as pure virtual.
	void Run();

private:
	std::mutex m_lock;
	Queue m_queue;
	XSemaphore m_signal;

	// if m_queue is empty, return m_emptyFn to avoid returning a dangling reference.
	Function m_emptyFn{ []() {} };
	Function m_fn;
};



#endif