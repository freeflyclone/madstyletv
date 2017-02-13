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

		mqtt->AddMessageListener("test/topic", [](XMqtt::Message m) {
			xprintf("TopicListener: '%.*s'\n", m->payloadlen, m->payload);
		});

		shape->AddChild(mqtt);
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
