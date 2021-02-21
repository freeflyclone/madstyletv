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
//#include "example.h"

#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_Z16    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           640               // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          0                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
#define HEIGHT_RATIO    20                // Defines the height ratio between the original frame to the new frame //
#define WIDTH_RATIO     10                // Defines the width ratio between the original frame to the new frame  //

class XGLDepthCam : public XGLTexQuad, public XThread {
public:
	XGLDepthCam() : XGLTexQuad(), XThread("DepthCamThread") {
		GenUShortZMap(WIDTH, HEIGHT);
	}

	void InitRealSense() {
		rs2_error* e{ nullptr };

		m_ctx = rs2_create_context(RS2_API_VERSION, &e);
		check_error(e);

		rs2_device_list* device_list = rs2_query_devices(m_ctx, &e);
		check_error(e);

		int dev_count = rs2_get_device_count(device_list, &e);
		check_error(e);
		xprintf("There are %d connected RealSense devices.\n", dev_count);
		if (0 == dev_count)
			return;

		// Get the first connected device
		// The returned object should be released with rs2_delete_device(...)
		rs2_device* dev = rs2_create_device(device_list, 0, &e);
		check_error(e);

		/* Determine depth value corresponding to one meter */
		m_one_meter = (uint16_t)(1.0f / get_depth_unit_value(dev));

		xprintf("one_meter: 1/%d\n", m_one_meter);

		// Create a pipeline to configure, start and stop camera streaming
		// The returned object should be released with rs2_delete_pipeline(...)
		pipeline = rs2_create_pipeline(m_ctx, &e);
		check_error(e);

		// Create a config instance, used to specify hardware configuration
		// The retunred object should be released with rs2_delete_config(...)
		rs2_config* config = rs2_create_config(&e);
		check_error(e);

		// Request a specific configuration
		rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
		check_error(e);

		// Start the pipeline streaming
		// The retunred object should be released with rs2_delete_pipeline_profile(...)
		rs2_pipeline_profile* pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
		if (e)
		{
			xprintf("The connected device doesn't support depth streaming!\n");
			exit(EXIT_FAILURE);
		}

		rs2_stream_profile_list* stream_profile_list = rs2_pipeline_profile_get_streams(pipeline_profile, &e);
		if (e)
		{
			xprintf("Failed to create stream profile list!\n");
			exit(EXIT_FAILURE);
		}

		rs2_stream_profile* stream_profile = (rs2_stream_profile*)rs2_get_stream_profile(stream_profile_list, 0, &e);
		if (e)
		{
			xprintf("Failed to create stream profile!\n");
			exit(EXIT_FAILURE);
		}

		rs2_stream stream; 
		rs2_format format; 
		int index; 
		int unique_id; 
		int framerate;

		rs2_get_stream_profile_data(stream_profile, &stream, &format, &index, &unique_id, &framerate, &e);
		if (e)
		{
			xprintf("Failed to get stream profile data!\n");
			exit(EXIT_FAILURE);
		}

		rs2_get_video_stream_resolution(stream_profile, &m_width, &m_height, &e);
		if (e)
		{
			xprintf("Failed to get video stream resolution data!\n");
			exit(EXIT_FAILURE);
		}
		int rows = m_height / HEIGHT_RATIO;
		int row_length = m_width / WIDTH_RATIO;
		int display_size = (rows + 1) * (row_length + 1);
		int buffer_size = display_size * sizeof(char);

		m_buffer = new char[display_size, sizeof(char)];
		xprintf("End of %s with no blow-ups.\n", __FUNCTION__);
	}

	void Run(void) {
		rs2_error* e{ nullptr };
		int rows = m_height / HEIGHT_RATIO;
		int row_length = m_width / WIDTH_RATIO;
		int display_size = (rows + 1) * (row_length + 1);
		int buffer_size = display_size * sizeof(char);

		InitRealSense();

		while (IsRunning()) {
			char* out = NULL;

			// This call waits until a new composite_frame is available
			// composite_frame holds a set of frames. It is used to prevent frame drops
			// The returned object should be released with rs2_release_frame(...)
			rs2_frame* frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, &e);
			check_error(e);

			// Returns the number of frames embedded within the composite frame
			int num_of_frames = rs2_embedded_frames_count(frames, &e);
			check_error(e);

			xprintf("%d frames\n", num_of_frames);

			int i;
			for (i = 0; i < num_of_frames; ++i)
			{
				// The retunred object should be released with rs2_release_frame(...)
				rs2_frame* frame = rs2_extract_frame(frames, i, &e);
				check_error(e);

				// Check if the given frame can be extended to depth frame interface
				// Accept only depth frames and skip other frames
				if (0 == rs2_is_frame_extendable_to(frame, RS2_EXTENSION_DEPTH_FRAME, &e))
				{
					rs2_release_frame(frame);
					continue;
				}

				/* Retrieve depth data, configured as 16-bit depth values */
				const uint16_t* depth_frame_data = (const uint16_t*)(rs2_get_frame_data(frame, &e));
				check_error(e);

				/* Print a simple text-based representation of the image, by breaking it into 10x5 pixel regions and approximating the coverage of pixels within one meter */
				out = m_buffer;
				int x, y, i;

				xprintf("Depth frame found\n");
				rs2_release_frame(frame);
			}

			rs2_release_frame(frames);

			std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		}
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

	void check_error(rs2_error* e)
	{
		if (e)
		{
			xprintf("rs_error was raised when calling %s(%s):\n", rs2_get_failed_function(e), rs2_get_failed_args(e));
			xprintf("    %s\n", rs2_get_error_message(e));
			throw XException("Unknown", 0, "check_error failed");
		}

	}

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

private:
	uint16_t m_one_meter{ 0 };
	rs2_context* m_ctx;
	char* m_buffer;
	rs2_pipeline* pipeline{ nullptr };
	int m_width; int m_height;

};

void ExampleXGL::BuildScene() {
	XGLDepthCam *depthCam;

	AddShape("shaders/zval_in_red", [&depthCam]() { depthCam = new XGLDepthCam(); return depthCam; });
	depthCam->attributes.diffuseColor = XGLColors::yellow;
	depthCam->model = glm::scale(glm::mat4(), glm::vec3(10.0f, 7.5f, 1.0f));

	depthCam->Start();
}
