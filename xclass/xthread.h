#ifndef XTHREAD_H
#define XTHREAD_H
#include <string>
#include <thread>

class XThread {
public:
	XThread(std::string n) : name(n) {};

	void Start() {
		t = std::thread(&XThread::Run, this);
	}

	virtual void Run() = 0;

	void WaitForJoin() {
		if (t.joinable())
			t.join();
	}

private:
	std::thread t;
	std::string name;
};

#endif
