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

		std::mt19937 e2(0xDEADC0ED);
		std::uniform_int_distribution<int> d2{ 1, 500 };

		while (IsRunning()){
			int nWrite;
			int randomNumber = d(e);
			int dly = d2(e2);
			nWrite = pcb->Write((unsigned char *)&randomNumber, sizeof(randomNumber));
			if (nWrite != sizeof(randomNumber))
				printf("pcb->Write() returned: %d\n", nWrite);
			//std::this_thread::sleep_for(std::chrono::duration<int, std::micro>(dly));
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

		std::mt19937 e2(0xDEADC0ED);
		std::uniform_int_distribution<int> d2{ 10, 50 };

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
	int seed = 0xC0EDFED0;
	XCircularBuffer cb(32768);

	Producer *p = new Producer(&cb, seed);
	Consumer *c = new Consumer(&cb, seed);

	c->Start();
	p->Start();

	std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(5000));

	c->Stop();
	std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1000));

	p->Stop();

	std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(1000));

	delete p;
	delete c;

	return 0;
}