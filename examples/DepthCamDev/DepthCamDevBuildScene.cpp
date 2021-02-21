/**************************************************************
** DepthCamDevBuildScene.cpp
**
** Intel's RealSense L515 LIDAR camera is some dope technology.
** Let's see how easily we can make use of the depth data.
** Piecemeal derive from 'rs-depth.c' sample in the RS2 SDK
**************************************************************/
#include "ExampleXGL.h"

#include <initguid.h>
#include <usbiodef.h>

#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_option.h>
#include <librealsense2/h/rs_frame.h>
#include "example.h"

#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_Z16    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           1024               // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          768                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
#define HEIGHT_RATIO    20                // Defines the height ratio between the original frame to the new frame //
#define WIDTH_RATIO     10                // Defines the width ratio between the original frame to the new frame  //

// The number of meters represented by a single depth unit
float get_depth_unit_value(const rs2_device* const dev)
{
	rs2_error* e = 0;
	rs2_sensor_list* sensor_list = rs2_query_sensors(dev, &e);
	check_error(e);

	int num_of_sensors = rs2_get_sensors_count(sensor_list, &e);
	check_error(e);

	float depth_scale = 0;
	int is_depth_sensor_found = 0;
	int i;
	for (i = 0; i < num_of_sensors; ++i)
	{
		rs2_sensor* sensor = rs2_create_sensor(sensor_list, i, &e);
		check_error(e);

		// Check if the given sensor can be extended to depth sensor interface
		is_depth_sensor_found = rs2_is_sensor_extendable_to(sensor, RS2_EXTENSION_DEPTH_SENSOR, &e);
		check_error(e);

		if (1 == is_depth_sensor_found)
		{
			depth_scale = rs2_get_option((const rs2_options*)sensor, RS2_OPTION_DEPTH_UNITS, &e);
			check_error(e);
			rs2_delete_sensor(sensor);
			break;
		}
		rs2_delete_sensor(sensor);
	}
	rs2_delete_sensor_list(sensor_list);

	if (0 == is_depth_sensor_found)
	{
		printf("Depth sensor not found!\n");
		exit(EXIT_FAILURE);
	}

	return depth_scale;
}

class XGLDepthCam : public XGLTexQuad {
public:
	XGLDepthCam() : XGLTexQuad() {
		GenUShortZMap(WIDTH, HEIGHT);
	}

	void GenUShortZMap(const int width, const int height) {
		GLuint texId;

		glGenTextures(1, &texId);
		GL_CHECK("Eh, something failed");
		glActiveTexture(GL_TEXTURE0 + numTextures);
		GL_CHECK("Eh, something failed");
		glBindTexture(GL_TEXTURE_2D, texId);
		GL_CHECK("Eh, something failed");
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		GL_CHECK("Eh, something failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		GL_CHECK("Eh, something failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		GL_CHECK("Eh, something failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		GL_CHECK("Eh, something failed");
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		GL_CHECK("Eh, something failed");
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, (void *)(nullptr));
		GL_CHECK("Eh, something failed");

		AddTexture(texId);
	}
};

void ExampleXGL::BuildScene() {
	XGLDepthCam *depthCam;

	AddShape("shaders/zval_in_red", [&depthCam]() { depthCam = new XGLDepthCam(); return depthCam; });
	depthCam->attributes.diffuseColor = XGLColors::yellow;
	depthCam->model = glm::scale(glm::mat4(), glm::vec3(10.0f, 7.5f, 1.0f));
}
