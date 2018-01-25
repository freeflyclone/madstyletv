/**************************************************************
** SmartThreadsBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

class DataThread : public XThread {
public:
	DataThread(std::string n) : XThread(n) {
		xprintf("DataThread::DataThread('%s')\n", Name().c_str());
	};
	virtual ~DataThread() {
		xprintf("DataThread::~DataThread('%s')\n", Name().c_str());
	};

	void Run(){
		while (IsRunning()) {
			xprintf("DataThread::Run('%s')\n", Name().c_str());
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(100));
		}
		xprintf("DataThread::Run('%s') falling through\n", Name().c_str());
	};
};

class AVPlayer : public XThread {
public:
	AVPlayer(std::string n) : XThread(n) {
		xprintf("AVPlayer::AVPlayer('%s')\n", Name().c_str());
	}
	virtual ~AVPlayer() {
		xprintf("AVPlayer::~AVPlayer('%s')\n", Name().c_str());
	}

	void Run() {
		dataThreads.push_back(XThreadCreate<DataThread>("DataThread1"));
		dataThreads.push_back(XThreadCreate<DataThread>("DataThread2"));
		dataThreads.push_back(XThreadCreate<DataThread>("DataThread3"));

		for (auto dt : dataThreads)
			dt->Start();

		while (IsRunning()) {
			xprintf("AVPlayer::Run('%s')\n", Name().c_str());
			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(200));
		}
		xprintf("AVPlayer::Run('%s') falling through\n", Name().c_str());
	}

	XThreadPool dataThreads;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	XThreadHandle avp = XThreadCreate<AVPlayer>("AVPlayerTest");

	AddThread(avp);

	avp->Start();
}
