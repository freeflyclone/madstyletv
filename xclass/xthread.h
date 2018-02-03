/*
** xthread.h
**
** XClass object for managing threads in C++ 11.
**
** There is a non-obvious situation that must be dealt with
** when using std::thread member variables AND attempting
** to start them from the constructor of the containing class.
**
** Refer to: 
**  https://rafalcieslak.wordpress.com/2014/05/16/c11-stdthreads-managed-by-a-designated-class/
**
** for details.  Basically: derived (from XThread) classes will not have valid vtable pointers
** until AFTER the constructor runs.  To cope with that, a Start() method is used, to be called AFTER
** the constructor has finished.  By doing so, the pure virtual Run() method will have been
** set up in the vtable by the derived class constructor, and can then be referenced here in the
** base class Start() method. Notice the move semantics in that function.
**
** The comments below are from the article referenced in the link above.
*/
#ifndef XTHREAD_H
#define XTHREAD_H
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>

class XThread {
public:
	/*
	** Explicitly using the default constructor to
	** underline the fact that it does get called
	*/
	XThread(std::string n) : name(n), isRunning(false), t() {};

	virtual ~XThread() {
		isRunning = false;
		if (t.joinable())
			t.join();
	}

	bool IsRunning() { return isRunning; }

	void Start() {
		isRunning = true;

		// This will start the thread. Notice move semantics!
		t = std::thread(&XThread::Run, this);
	}

	void Stop() {
		isRunning = false;
	}

	bool WaitForStop() {
		Stop();
		return WaitForJoin();
	}

	virtual void Run() = 0;

	bool WaitForJoin() {
		if (t.joinable()) {
			t.join();
			return true;
		}
		else 
			return false;
	}

	std::string Name() { return name; }

private:
	std::thread t;
	std::string name;
	bool isRunning;
};

typedef std::shared_ptr<XThread> XThreadHandle;
typedef std::vector<XThreadHandle> XThreadPool;

template <typename T>
XThreadHandle XThreadCreate(std::string n) {
	return std::make_shared<T>(n);
}

class XSemaphore {
public:
	// Initially, this semaphore must be notify()'d before a wait() will be satisfied
	XSemaphore(int c = 0) : count(c) {};

	void operator()(int c) {
		std::unique_lock<std::mutex> lock(mutex);
		count = c;
	}

	void notify(int n = 1) {
		std::unique_lock<std::mutex> lock(mutex);
		count += n;
		cv.notify_one();
	}

	void wait(int n = 1) {
		std::unique_lock<std::mutex> lock(mutex);
		waitNum = n;
		cv.wait(lock, [this]{ return (count > waitNum); });

		count -= waitNum;
	}
	
	bool wait_for(int dlyInMillis, int n = 1) {
		std::unique_lock<std::mutex> lock(mutex);
		waitForNum = n;
		bool retVal = cv.wait_for(lock, std::chrono::milliseconds(dlyInMillis), [this] { return (count >= waitForNum); });

		// if retVal is true, it means we didn't time out.
		if (retVal)
			count -= waitForNum;

		return retVal;
	}

	unsigned int get_count() { 
		std::unique_lock<std::mutex> lock(mutex); 
		return count;
	}

private:
	unsigned int count;
	unsigned int waitNum;
	unsigned int waitForNum;
	std::mutex mutex;
	std::condition_variable cv;
};
#endif
