/**************************************************************
** ImuDevBuildScene.cpp
**
** Demonstrate instantiation of my IMU sensor object in the
** XGL framework.
**************************************************************/
#include "ExampleXGL.h"
#include "graph.h"

XGLShape *shape;
XGLGraph *gyroXgraph;
XGLGraph *gyroYgraph;
XGLGraph *gyroZgraph;

XUartAscii *xuart;

float gyro[3];
float accel[3];
float mag[3];

// Ascii HexToShort(), a nibble at a time
short HexToShort(unsigned char *hex) {
	int idx;
	short total = 0;

	// each variable is expressed as a signed short,
	// LSB first.  So do MSB first.
	for (int i=2; i<4; i++) {
		if (hex[i] >= '0' && hex[i] <= '9')
			idx = hex[i] - '0';
		else
			idx = hex[i] - 'A' + 10;

		total += idx;
		total <<= 4;
	}
	for (int i=0; i<2; i++) {
		if (hex[i] >= '0' && hex[i] <= '9')
			idx = hex[i] - '0';
		else
			idx = hex[i] - 'A' + 10;

		total += idx;

		// don't over-rotate the final nibble
        if (i==0)
            total <<= 4;
	}
	return total;
}

void ExampleXGL::BuildScene() {
	glm::mat4 translate;

	glm::vec3 cameraPosition(0, -0.01, 20);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection,	cameraUp);


	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("IMU Plane", false);

    CreateShape("shaders/000-attributes", [&]() { gyroXgraph = new XGLGraph(); return gyroXgraph;});
    gyroXgraph->attributes.diffuseColor = XGLColors::green;

    CreateShape("shaders/000-attributes", [&]() { gyroYgraph = new XGLGraph(); return gyroYgraph;});
    gyroYgraph->attributes.diffuseColor = XGLColors::red;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, 5, 0));
	gyroYgraph->model = translate;

    CreateShape("shaders/000-attributes", [&]() { gyroZgraph = new XGLGraph(); return gyroZgraph;});
    gyroZgraph->attributes.diffuseColor = XGLColors::blue;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -5, 0));
	gyroZgraph->model = translate;

    shape->AddChild(gyroXgraph);
    shape->AddChild(gyroYgraph);
    shape->AddChild(gyroZgraph);

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([&](unsigned char *line){
			short imuData[9];
			for (int i=0; i<9; i++)
				imuData[i] = HexToShort(line+(i*5));
			
			for (int i=0; i<3; i++) 
				gyro[i] = (float)imuData[i] / 32767.0f * 2000.0;

			gyroXgraph->NewValue(gyro[0]/100.0f);
			gyroYgraph->NewValue(gyro[1]/100.0f);
			gyroZgraph->NewValue(gyro[2]/100.0f);
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
