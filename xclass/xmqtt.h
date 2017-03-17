#ifndef XMQTT_H
#define XMQTT_H

#include <string>
#include <sstream>
#include <stdexcept>
#include "mosquitto.h"
#include "xobject.h"
#include "xthread.h"
#include "xutils.h"

class XMqtt : public XObject {
public: 
	// map Mosquitto types to more generic names
    typedef mosquitto *Mqtt;
    typedef const mosquitto_message *Message;

	// define Listener functions so lambda's are easier to grok
	typedef std::function<void(Mqtt, XMqtt*, int)> ConnectListener;
	typedef std::function<void(Mqtt, XMqtt*, int)> DisconnectListener;
	typedef std::function<void(Message)> MessageListener;

	typedef std::vector<ConnectListener> ConnectListeners;
	typedef std::vector<DisconnectListener> DisconnectListeners;

	// A list of MessageListener objects that get called regardless of topic
	typedef std::vector<MessageListener> MessageListeners;

	// Topic-specific MessageListener objects
	typedef std::map<std::string, MessageListeners> TopicMessageListeners;

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

    static void ConnectCallback(Mqtt, void *, int);
    static void DisconnectCallback(Mqtt, void *, int);
    static void MessageCallback(Mqtt, void *, Message);

	void AddConnectListener(ConnectListener);
	void AddDisconnectListener(DisconnectListener);
	void AddMessageListener(MessageListener);
	int AddMessageListener(std::string, MessageListener);

	int Publish(std::string, int, void *, int);

	std::string ConnectState(int c);

private:
    std::string host;
    int port;
    Mqtt mq;
    LoopThread *loopThread;
	
	ConnectListeners connectListeners;
	DisconnectListeners disconnectListeners;
	TopicMessageListeners topicMessageListeners;
	MessageListeners listeners;
};

#endif // XMQTT_H
