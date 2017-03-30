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

	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("IMU Plane", false);

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([](unsigned char *line){
			short imuData[9];
			for (int i=0; i<9; i++)
				imuData[i] = HexToShort(line+(i*5));
			
			printf("%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
				imuData[0], imuData[1],imuData[2],
				imuData[3], imuData[4],imuData[5],
				imuData[6], imuData[7],imuData[8]);
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
