#ifndef XUARTASCII_H
#define XUARTASCII_H

#include "xclasses.h"
#include "xuart.h"

class XUartAscii : public XObject, public XUart {
public:
    typedef std::function<void(unsigned char *)> Listener;
	typedef std::vector<Listener> Listeners;

	class ReadThread : public XThread {
	public:
        typedef enum { NotSynced, Synced } ReadState;
		ReadThread(XUartAscii &);
		~ReadThread();
		void Run();

	private:
        unsigned char cp;
        ReadState state;
		XUartAscii &pAscii;
        unsigned char buffer[2048];
        unsigned char *insertPoint;
	};

	class WriteThread : public XThread {
	public:
		WriteThread(XUartAscii &);
		~WriteThread();
		void Run();
		bool WriteMessage();

	private:
		XUartAscii &pAscii;
	};

	XUartAscii(std::string);
	~XUartAscii();
	void AddListener(Listener);

private:
	ReadThread *rThread;
	WriteThread *wThread;
	Listeners listeners;
};


#endif
