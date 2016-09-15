/**************************************************************
** main.cpp
**
** ThreadsTest project.  Actually tests XThread, XSemaphore
** and XCircularBuffer.
**
** Implements a Producer/Consumer pattern with a circular
** buffer.  Both the Producer and Consumer run random number
** sequences based on the same key. Thus the consumer can
** verify the Producer's "random" data in the circular buffer.
*************************************************************/
#include <stdio.h>
#include <xutils.h>
#include <xthread.h>
#include <xcircularbuffer.h>
#include <chrono>
#include <random>

class Producer : public XThread {
public:
	Producer(XCircularBuffer *cb, int seed) : XThread("Producer"), pcb(cb), rngSeed(seed) {};

	void Run() {
		std::mt19937 e(rngSeed);
		std::uniform_int_distribution<int> d;

		std::mt19937 e2(0xFEEDC0ED);
		std::uniform_int_distribution<int> d2{ 1, 500 };

		while (IsRunning()){
			int nWrite;
			int randomNumber = d(e);
			int dly = d2(e2);
			nWrite = pcb->Write((unsigned char *)&randomNumber, sizeof(randomNumber));
		}
	}

	XCircularBuffer *pcb;
	int rngSeed;
};

class Consumer : public XThread {
public:
	Consumer(XCircularBuffer *cb, int seed) : XThread("Consumer"), pcb(cb), rngSeed(seed) {};

	void Run() {
		std::mt19937 e(rngSeed);
		std::uniform_int_distribution<int> d;

		std::mt19937 e2(0xC0EDBEEF);
		std::uniform_int_distribution<int> d2{ 40, 80 };

		while (IsRunning()){
			int dly = d2(e2);
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(dly));
			printf("Count(): %d\n", pcb->Count());

			while (pcb->Count()) {
				int randomNumber = d(e);
				int producedNumber;
				pcb->Read((unsigned char *)&producedNumber, sizeof(producedNumber));

				if (randomNumber != producedNumber)
					printf("Unexpected number read, %d vs %d\n", producedNumber, randomNumber);
			}
		}
	}

	XCircularBuffer *pcb;
	int rngSeed;
};

int main(int argc, char *argv[]) {
	try {
		XCircularBuffer cb(0x10000);
		Producer *p;
		Consumer *c;

		p = new Producer(&cb, 0xC0EDFED0);
		c = new Consumer(&cb, 0xC0EDFED0);

		p->Start();
		c->Start();

		std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(2000));
		
		delete p;
		delete c;
	}
	catch (std::runtime_error e) {
		printf("Exception: %s\n", e.what());
	}
}