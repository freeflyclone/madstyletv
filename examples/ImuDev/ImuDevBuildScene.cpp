/**************************************************************
** ImuDevBuildScene.cpp
**
** Demonstrate instantiation of my IMU sensor object in the
** XGL framework.
**************************************************************/
#include "ExampleXGL.h"
#include "graph.h"

XGLShape *shape;
XGLGraph *graph;

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
	glm::vec3 cameraPosition(0, -0.01, 20);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection,	cameraUp);


	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("IMU Plane", false);

    CreateShape("shaders/000-simple", [&]() { graph = new XGLGraph(); return graph;});
    graph->attributes.diffuseColor = XGLColors::green;
    shape->AddChild(graph);

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([](unsigned char *line){
			short imuData[9];
			for (int i=0; i<9; i++)
				imuData[i] = HexToShort(line+(i*5));
			
			for (int i=0; i<3; i++) {
				gyro[i] = (float)imuData[i] / 32767.0f * 2000.0;
			}
/*
			printf("%s", line);

			printf("%6.3f,%6.3f,%6.3f,%d,%d,%d,%d,%d,%d\n", 
				gyro[0], gyro[1], gyro[2],
				imuData[3], imuData[4],imuData[5],
				imuData[6], imuData[7],imuData[8]);
*/
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
