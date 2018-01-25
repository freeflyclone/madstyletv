/**************************************************************
** main.cpp
**
** SemaphoreTest project: simple unit test for XSemaphore.
**
** Specifically, testing for Producer/Consumer counting semaphore
** semantics.
*************************************************************/
#include <stdio.h>
#include <xutils.h>
#include <xthread.h>
#include <xcircularbuffer.h>
#include <chrono>
#include <random>
#include <stdarg.h>

XSemaphore freeCount(4), usedCount;
std::mutex pMutex;

void MyPrintf(char *fmt, ...) {
	std::lock_guard<std::mutex> lock(pMutex);

	va_list list;
	va_start(list, fmt);

	vprintf(fmt, list);
}

class ConsumeThread : public XThread {
public:
	ConsumeThread() : XThread("ConsumeThread") {};

	void Run() {
		printf("ConsumeThread::Run() started\n");
		while (IsRunning()) {
			usedCount.wait();
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			freeCount.notify();
			MyPrintf("ConsumeThread::Run(): %d\n", usedCount.get_count());
		}
		MyPrintf("ConsumeThread::Run() ended\n");
	}
};

class ProduceThread : public XThread {
public:
	ProduceThread() : XThread("ProduceThread") {};

	void Run() {
		MyPrintf("ProduceThread::Run() started\n");
		while (IsRunning()) {
			//freeCount.wait();
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			MyPrintf("ProduceThread::Run(), about to produce\n");
			usedCount.notify();
		}
		MyPrintf("ProduceThread::Run() ended\n");
		usedCount.notify();
	}
};

ConsumeThread ct;
ProduceThread pt;

int main(int argc, char *argv[]) {
	bool done{ false };

	MyPrintf("Starting...\n");
	pt.Start();
	ct.Start();

	std::this_thread::sleep_for(std::chrono::milliseconds(2000));

	MyPrintf("Done.\n", "Test");

	ct.Stop();
	pt.Stop();
}