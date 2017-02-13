#ifndef XMQTT_H
#define XMQTT_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <mosquitto.h>
#include "xobject.h"
#include "xthread.h"
#include "xutils.h"

class XMqtt : public XObject {
public: 
    typedef mosquitto *Mosquitto;
    typedef const mosquitto_message *Message;

    class LoopThread : public XThread {
    public:
        LoopThread(XMqtt &);
        ~LoopThread();
        void Run();

    private:
        XMqtt &pMqtt;
    };

    XMqtt(std::string h="localhost", int p=1883);
	~XMqtt();

    static void ConnectCallback(Mosquitto, void *, int);
    static void MessageCallback(Mosquitto, void *, Message);

	int Read(unsigned char *b, int size);
	int Write(unsigned char *b, int size);

private:
    std::string host;
    int port;
    Mosquitto mosq;
    LoopThread *loopThread;
};

#endif // XMQTT_H
