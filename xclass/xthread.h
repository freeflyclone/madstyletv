#ifndef XTHREAD_H
#define XTHREAD_H
#include <string>
#include <thread>

class xthread {
public:
	xthread(std::string n) : name(n) {};

	void Start(xthread& foo) {
		t = std::thread(&xthread::Run, this);
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
