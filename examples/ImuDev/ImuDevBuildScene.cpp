/**************************************************************
** ImuDevBuildScene.cpp
**
** Demonstrate instantiation of my IMU sensor object in the
** XGL framework.
**************************************************************/
#include "ExampleXGL.h"
#include "graph.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

extern "C" {
#include "MadgwickAHRS/MadgwickAHRS.h"
}

XGLShape *shape;
XGLGraph *gyroXgraph, *gyroYgraph,*gyroZgraph;
XGLGraph *accelXgraph, *accelYgraph,*accelZgraph;
XGLGraph *magXgraph, *magYgraph,*magZgraph;
XGLGraph *gyroRateGraph,*accelRateGraph;

XUartAscii *xuart;

float gyroCal[3] = {0.01811, -0.054863, -0.054863};

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

	AddShape("shaders/diffuse", [&](){ shape = new XGLCube(); return shape; });
	shape->SetName("IMU Plane", false);

    AddShape("shaders/000-attributes", [&]() { gyroXgraph = new XGLGraph(); return gyroXgraph;});
    gyroXgraph->attributes.diffuseColor = XGLColors::green;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, 15, 0));
	gyroXgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { gyroYgraph = new XGLGraph(); return gyroYgraph;});
    gyroYgraph->attributes.diffuseColor = XGLColors::red;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, 10, 0));
	gyroYgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { gyroZgraph = new XGLGraph(); return gyroZgraph;});
    gyroZgraph->attributes.diffuseColor = XGLColors::blue;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, 5, 0));
	gyroZgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { accelXgraph = new XGLGraph(); return accelXgraph;});
    accelXgraph->attributes.diffuseColor = XGLColors::yellow;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -5, 0));
	accelXgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { accelYgraph = new XGLGraph(); return accelYgraph;});
    accelYgraph->attributes.diffuseColor = XGLColors::cyan;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -10, 0));
	accelYgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { accelZgraph = new XGLGraph(); return accelZgraph;});
    accelZgraph->attributes.diffuseColor = XGLColors::magenta;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -15, 0));
	accelZgraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { gyroRateGraph = new XGLGraph(); return gyroRateGraph;});
    gyroRateGraph->attributes.diffuseColor = XGLColors::cyan;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -20, 0));
	gyroRateGraph->model = translate;

    AddShape("shaders/000-attributes", [&]() { accelRateGraph = new XGLGraph(); return accelRateGraph;});
    accelRateGraph->attributes.diffuseColor = XGLColors::white;
    translate = glm::translate(glm::mat4(), glm::vec3(0.0, -25, 0));
	accelRateGraph->model = translate;

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([&](unsigned char *line){
			static long int count = 0;
			const long int maxCount = 250;
			short imuData[9];
			float y,p,r;

			for (int i=0; i<9; i++)
				imuData[i] = HexToShort(line+(i*5));
			
			for (int i=0; i<3; i++) {
				gyro[i] = (((float)imuData[i] / 32767.0f * 2000.0f) / 180.f * M_PI);
				accel[i] = (float)imuData[i+3] / 16384.0f * 4.0f;
				mag[i] = (float)imuData[i+6] / 16384.0f;
			}

			gyroXgraph->NewValue(gyro[0]);
			gyroYgraph->NewValue(gyro[1]);
			gyroZgraph->NewValue(gyro[2]);

			accelXgraph->NewValue(accel[0]);
			accelYgraph->NewValue(accel[1]);
			accelZgraph->NewValue(accel[2]);

			gyroRateGraph->NewValue(gyroRateChange/10.0f);
			accelRateGraph->NewValue(accelRateChange);

			if (count < maxCount) {
				for (int i=0; i<3; i++)
					gyroCal[i] += gyro[i];
			}
			else if(count == maxCount) {
				for (int i=0; i<3; i++)
					gyroCal[i] /= (float)maxCount;
			}
			else {
				gyro[0] -= gyroCal[0];
				gyro[1] -= gyroCal[1];
				gyro[2] -= gyroCal[2];

				//MadgwickAHRSupdate(gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2], mag[0], mag[1], mag[2]);
				MadgwickAHRSupdateIMU(gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2]);

				glm::quat myQuat = glm::quat((double)q0, (double)q1, (double)q2, (double)q3);
				glm::mat4 rotate = glm::toMat4(myQuat);

				shape->model = rotate;
			}
			count++;
		});

		shape->AddChild(xuart);
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
