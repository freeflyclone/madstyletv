#ifndef XDISPATCHQ_H
#define XDISPATCHQ_H

#include "xclasses.h"
#include "xlog.h"
#include <queue>
/**
* Simple queue to execute async Function objects on alternate thread(s)
* Intended for executing lambda functions, where "capturing" of in-scope
* variables is possible.
*
* Post() adds a Function to the back of the Queue
* Remove() extracts a Function from the front of the Queue
*
* Run() executes a timed wait_for() on an XSemaphore(), and if "true" is
* returned it then Remove()'s a Function and runs it in the thread.
* Default constructor starts thread.  Constructor w/name does NOT spawn a thread.
* 
* Use of Remove() in multiple threads simultaneously is presently undefined.
*/
class XDispatchQueue {
public:
	typedef std::function<void()> Function;
	typedef std::queue<Function> Queue;

	XDispatchQueue();
	XDispatchQueue(const char* name);
	~XDispatchQueue();

	void Post(Function fn);
	Function& Remove();
	
	void Run();

private:
	std::string m_name;
	std::atomic_bool m_begin{ false };
	bool m_isRunning{ true };
	std::thread m_thread;

	std::mutex m_lock;
	Queue m_queue;
	XSemaphore m_signal;

	// if m_queue is empty, return m_emptyFn to avoid returning a dangling reference.
	// (Sub-optimal, executing an empty lambda wastes CPU cycles.  But not TOO many.)
	Function m_emptyFn{ []() {} };
	Function m_fn;
};



#endif