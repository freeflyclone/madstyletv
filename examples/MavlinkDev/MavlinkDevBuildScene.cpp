/**************************************************************
** MavlinkDevBuildScene.cpp
**
** Demonstrate instantiation of XMavlink object in the
** XGL framework.  Add an XMavlink::Listener function for the
** MAVLINK_MSG_ID_ATTITUDE message and take the yaw,pitch and
** roll, build it into a model transformation matrix, and
** apply it to a shape.  Thus the IMU sensor is visualized
** in real-time from a MAVLINK compatible flight controller.
**************************************************************/
#include "ExampleXGL.h"
#include "xmavlink.h" 
#include "xuartascii.h"
//#include "xftdi.h"

XMavlink *mavlink;
//XFtdi *ftdi;
XUartAscii *ascii;

XGLShape *shape;

void ExampleXGL::BuildScene() {

	AddShape("shaders/specular", [&](){shape = new XGLSphere(1.0, 64); return shape; });
	shape->model = glm::translate(glm::mat4(), glm::vec3(10, 0, 0));
	shape->attributes.diffuseColor = XGLColors::green;

	AddShape("shaders/specular", [&](){shape = new XGLSphere(1.0, 64); return shape; });
	shape->model = glm::translate(glm::mat4(), glm::vec3(0, 10, 0));
	shape->attributes.diffuseColor = XGLColors::red;

	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("Mavlink Plane", false);

	try {
		// presently, XMavlink is derived from XUart.  The device name
		// of the serial port specifies which UART our MAVLINK device is connected to.
		// This should come from the configuration file.
        //mavlink = new XMavlink("\\\\.\\COM17");
        mavlink = new XMavlink("/dev/ttyACM0");

		// We're going to be adding an XMavlink::Listener for this shape, which is called
		// in the XMavlink::ReceiveThread (ie: not this thread) context.  Depending on
		// the timing of application shutdown, it's possible that "shape" will be deleted
		// as the Listener is being called, and the Listener will crash.  By making
		// "mavlink" an XObject child of "shape", it will get deleted first, thus preventing
		// an invalid callback from occuring.
        shape->AddChild(mavlink);

		// add a XMavlink::Listener function for ATTITUDE messages, that will move "shape" accordingly
		mavlink->AddListener(MAVLINK_MSG_ID_ATTITUDE, [&](mavlink_message_t msg){
			mavlink_attitude_t attitude;
			mavlink_msg_attitude_decode(&msg, &attitude);

			// build a rotation matrix out of yaw, pitch, and roll from attitude message
			glm::mat4 yaw = glm::rotate(glm::mat4(), -attitude.yaw, glm::vec3(0, 0, 1));
			glm::mat4 pitch = glm::rotate(glm::mat4(), -attitude.pitch, glm::vec3(0, 1, 0));
			glm::mat4 roll = glm::rotate(glm::mat4(), attitude.roll, glm::vec3(1, 0, 0));
			glm::mat4 rotate = yaw * pitch * roll;

			// scale the cube to make it look like a plane
			glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(8, 4, 0.1));

			// apply combined rotation and scale to the shape's model matrix
			shape->model = rotate * scale;
		});

	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
