// threads.cpp : simple console application that tests xthread
//
// A base class an std::thread member is problematic on OSX
// with using a destructor, so the function WaitForJoin()
// is called in classes derived from xthread as workaround for now.
// ----------------------------------------------------------------
#include "xthread.h"
#include "xutils.h"

class TestThread : public xthread {
public:
	TestThread(std::string n) : xthread(n) {
		xthread::Start(*this);
	};

	~TestThread() {
		xthread::WaitForJoin();
	}

	void Run() {
		printf(("TestThread runs --pass\n"));
	}
};

int main(int argc, char* argv[])
{
	xprintf("This is a test\n");
	TestThread tt("TestThread");
	return 0;
}

