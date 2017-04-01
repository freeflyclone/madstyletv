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

XUartAscii *xuart;

float gyro[3];
float accel[3];
float mag[3];

void QuatToEuler(float q1, float q2, float q3, float q4, float& pitch, float& yaw, float& roll) {
	float w = q1;
	float x = q2;
	float y = q3;
	float z = q4;

	pitch = atan2(2*(y*z + w*x), w*w - x*x - y*y + z*z);
	yaw = asin(-2*(x*z - w*y));
	roll = atan2(2*(x*y + w*z), w*w + x*x - y*y + z*z);
}

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

	glm::vec3 cameraPosition(0, -0.1, 40);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection,	cameraUp);


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

	try {
		xuart = new XUartAscii("/dev/ttyUSB0");
		xuart->AddListener([&](unsigned char *line){
			static long int count = 0;
			short imuData[9];
			float y,p,r;

			for (int i=0; i<9; i++)
				imuData[i] = HexToShort(line+(i*5));
			
			for (int i=0; i<3; i++) {
				//gyro[i] = (float)imuData[i] / 32767.0f * 2000.0;
				//accel[i] = (float)imuData[i+3] / 32767.0f * 32.0;
				gyro[i] = (float)imuData[i] / 32767.0f * 36.0f;
				accel[i] = (float)imuData[i+3] / 32767.0f;
			}

			gyroXgraph->NewValue(gyro[0]/100.0f);
			gyroYgraph->NewValue(gyro[1]/100.0f);
			gyroZgraph->NewValue(gyro[2]/100.0f);


			accelXgraph->NewValue(accel[0]);
			accelYgraph->NewValue(accel[1]);
			accelZgraph->NewValue(accel[2]);

			MadgwickAHRSupdateIMU(gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2]);
			QuatToEuler(q0, q1, q2, q3, p, y, r);

            // build a rotation matrix out of yaw, pitch, and roll from attitude message
            glm::mat4 yaw = glm::rotate(glm::mat4(), y, glm::vec3(0, 0, 1));
            glm::mat4 pitch = glm::rotate(glm::mat4(), p, glm::vec3(0, 1, 0));
            glm::mat4 roll = glm::rotate(glm::mat4(), r, glm::vec3(1, 0, 0));
            glm::mat4 rotate = yaw * pitch * roll;


			shape->model = yaw * pitch * roll;
		});
	}
	catch (std::runtime_error e) {
		xprintf("Well that didn't work out: %s\n", e.what());
	}
}
