#ifndef XMQTT_H
#define XMQTT_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <mosquitto.h>
#include "xutils.h"

class XMqtt {
public: 
	XMqtt();
	~XMqtt();

	int Read(unsigned char *b, int size);
	int Write(unsigned char *b, int size);

private:
};

#endif // XMQTT_H
