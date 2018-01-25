/**************************************************************
** MqttDevBuildScene.cpp
**
** Demonstrate instantiation of XMqtt object in the
** XGL framework.
**************************************************************/
#include "ExampleXGL.h"
#include "xmqtt.h" 

XMqtt *mqtt;
XGLShape *shape;

void ExampleXGL::BuildScene() {

	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("Mqtt Plane", false);

	try {
        mqtt = new XMqtt();

		mqtt->AddConnectListener([](XMqtt::Mqtt m, XMqtt *p, int code) {
			xprintf("Instance %s, connection: %s\n", p->Name().c_str(), mqtt->ConnectState(code).c_str());
		});

		mqtt->AddDisconnectListener([](XMqtt::Mqtt m, XMqtt *p, int code) {
			xprintf("Instance %s, disconnect: %d\n", p->Name().c_str());
		});

		mqtt->AddMessageListener("test/topic", [](XMqtt::Message m) {
			xprintf("TopicListener: '%.*s'\n", m->payloadlen, m->payload);
		});

		shape->AddChild((XGLShape*)mqtt);
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
