/**************************************************************
** ImuDevBuildScene.cpp
**
** Demonstrate instantiation of my IMU sensor object in the
** XGL framework.
**************************************************************/
#include "ExampleXGL.h"
#include "xuartascii.h" 

XGLShape *shape;
XUartAscii *xuart;

void ExampleXGL::BuildScene() {

	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("IMU Plane", false);

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([](unsigned char *line){
			printf("%s", line);
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
