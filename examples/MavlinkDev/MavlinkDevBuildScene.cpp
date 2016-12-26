/**************************************************************
** MavlinkDevBuildScene.cpp
**
** Demonstrate instantiation of XMavlink object in the
** XGL framework
**************************************************************/
#include "ExampleXGL.h"
#include "xmavlink.h"

XMavlink *mavlink;

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&shape](){ shape = new XGLTriangle(); return shape; });

	try {
		// presently, XMavlink is derived from XUart.  The device name
		// of the serial port specifies which UART our MAVLINK device is connected to.
		// This should come from the configuration file.
		mavlink = new XMavlink("\\\\.\\COM17");

		// add a XMavlink::Listener function for ATTITUDE messages, that will move "shape" accordingly
		mavlink->AddListener(MAVLINK_MSG_ID_ATTITUDE, [&shape](mavlink_message_t msg){
			xprintf("Listener attitude\n");
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
