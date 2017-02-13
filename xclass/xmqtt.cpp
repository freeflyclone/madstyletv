#include "xmqtt.h"

XMqtt::XMqtt() {
	mosquitto_lib_init();
}

XMqtt::~XMqtt() {
}

int XMqtt::Read(unsigned char *b, int size) {
}

int XMqtt::Write(unsigned char *b, int size) {
}
