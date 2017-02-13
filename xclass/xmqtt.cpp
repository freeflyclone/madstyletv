#include "xmqtt.h"

XMqtt::LoopThread::LoopThread(XMqtt &pm) : XThread("XMqtt::LoopThread"), pMqtt(pm) {
    Start();
}

XMqtt::LoopThread::~LoopThread() {
    Stop();
}

void XMqtt::LoopThread::Run() {
    while (IsRunning()) {
        mosquitto_loop(pMqtt.mosq, -1, 1);
    }
}

XMqtt::XMqtt(std::string h, int p) : host(h), port(p), loopThread(NULL) {
    SetName("XMqtt");

	mosquitto_lib_init();

    if (!(mosq = mosquitto_new(XObject::Name().c_str(), true, this)))
        throw std::runtime_error("Failed creating a new mosquitto instance" + Name());

    mosquitto_connect_callback_set(mosq, ConnectCallback);
    mosquitto_message_callback_set(mosq, MessageCallback);

    if (mosquitto_connect(mosq, host.c_str(), port, 10) != MOSQ_ERR_SUCCESS)
        throw std::runtime_error("Failed to connect to host '" + host + "':" + std::to_string(port));

    if (mosquitto_subscribe(mosq, NULL, "test/topic", 0) != MOSQ_ERR_SUCCESS)
        throw std::runtime_error("Failed to subscribe to 'test/topic'");

    loopThread = new LoopThread(*this);
}

XMqtt::~XMqtt() {
}

void XMqtt::ConnectCallback(Mosquitto mosq, void *obj, int result) {
    XMqtt *p = (XMqtt *)obj;

    xprintf("ConnectCallback(): %s, rc=%d\n", p->Name().c_str(), result);
    // do a Listener callback here
}

void XMqtt::MessageCallback(Mosquitto mosq, void *obj, Message msg) {
    XMqtt *p = (XMqtt *)obj;

    xprintf("MessageCallback(): '%.*s' for topic '%s'\n", msg->payloadlen, (char *)msg->payload, msg->topic);
    // do a Listener callback here
}

int XMqtt::Read(unsigned char *b, int size) {
}

int XMqtt::Write(unsigned char *b, int size) {
}
