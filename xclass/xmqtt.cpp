#include "xmqtt.h"

XMqtt::LoopThread::LoopThread(XMqtt &pm) : XThread("XMqtt::LoopThread"), pMqtt(pm) {
    Start();
}

XMqtt::LoopThread::~LoopThread() {
    Stop();
}

void XMqtt::LoopThread::Run() {
    while (IsRunning()) {
        mosquitto_loop(pMqtt.mq, -1, 1);
    }
}

XMqtt::XMqtt(std::string h, int p) : host(h), port(p), loopThread(NULL) {
    SetName("XMqtt_");

	mosquitto_lib_init();

    if (!(mq = mosquitto_new(XObject::Name().c_str(), true, this)))
        throw std::runtime_error("Failed creating a new mosquitto instance " + Name());

    mosquitto_connect_callback_set(mq, ConnectCallback);
    mosquitto_message_callback_set(mq, MessageCallback);

    if (mosquitto_connect(mq, host.c_str(), port, 10) != MOSQ_ERR_SUCCESS)
        throw std::runtime_error("No connection to host '" + host + "':" + std::to_string(port));

    loopThread = new LoopThread(*this);
}

XMqtt::~XMqtt() {
	if (loopThread)
		delete loopThread;

	mosquitto_disconnect(mq);
	mosquitto_destroy(mq);
}

void XMqtt::ConnectCallback(Mqtt mq, void *obj, int result) {
    XMqtt *p = (XMqtt *)obj;

    xprintf("ConnectCallback(): %s, rc=%d\n", p->Name().c_str(), result);
    // do a MessageListener callback here
}

void XMqtt::MessageCallback(Mqtt mq, void *obj, Message msg) {
    XMqtt *p = (XMqtt *)obj;

    for (auto fn : p->topicMessageListeners[msg->topic])
        fn(msg);
	for (auto fn : p->listeners)
		fn(msg);
}

int XMqtt::AddMessageListener(std::string topic, MessageListener l) {
	int retVal = mosquitto_subscribe(mq, NULL, topic.c_str(), 0);
	if (retVal == MOSQ_ERR_SUCCESS)
		topicMessageListeners[topic].push_back(l);
	return retVal;
}

void XMqtt::AddMessageListener(MessageListener l) {
	listeners.push_back(l);
}
