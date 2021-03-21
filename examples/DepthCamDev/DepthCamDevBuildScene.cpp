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

#include "DepthCloud.h"

//#include "example.h"

#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_ANY    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           1024              // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          768                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    -1                // Defines the stream index, used for multiple streams of the same type //
#define HEIGHT_RATIO    20                // Defines the height ratio between the original frame to the new frame //
#define WIDTH_RATIO     10                // Defines the width ratio between the original frame to the new frame  //

class XGLDepthCam : public XGLTexQuad, public XThread {
public:
	static const int m_numFrames{ 4 };

	XGLDepthCam() : XGLTexQuad(), XThread("DepthCamThread") {
		GenUShortZMap(WIDTH, HEIGHT);

		InitRealSense();
	}

	~XGLDepthCam() {
		xprintf("%s\n", __FUNCTION__);
	}

	void InitRealSense() {
		rs2_error* e{ nullptr };

		m_ctx = rs2_create_context(RS2_API_VERSION, &e);
		check_error(e);

		m_device_list = rs2_query_devices(m_ctx, &e);
		check_error(e);

		m_dev_count = rs2_get_device_count(m_device_list, &e);
		check_error(e);
		xprintf("There are %d connected RealSense devices.\n", m_dev_count);
		if (0 == m_dev_count)
			return;

		// Get the first connected device
		// The returned object should be released with rs2_delete_device(...)
		m_dev = rs2_create_device(m_device_list, 0, &e);
		check_error(e);

		/* Determine depth value corresponding to one meter */
		m_one_meter = (uint16_t)(1.0f / get_depth_unit_value(m_dev));

		xprintf("one_meter: 1/%d\n", m_one_meter);

		// Create a pipeline to configure, start and stop camera streaming
		// The returned object should be released with rs2_delete_pipeline(...)
		m_pipeline = rs2_create_pipeline(m_ctx, &e);
		check_error(e);

		// Create a config instance, used to specify hardware configuration
		// The retunred object should be released with rs2_delete_config(...)
		m_config = rs2_create_config(&e);
		check_error(e);

		// Request a specific configuration
		rs2_config_enable_stream(m_config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
		check_error(e);

		// Start the pipeline streaming
		// The retunred object should be released with rs2_delete_pipeline_profile(...)
		m_pipeline_profile = rs2_pipeline_start_with_config(m_pipeline, m_config, &e);
		if (e)
		{
			xprintf("The connected device doesn't support depth streaming!\n");
			exit(EXIT_FAILURE);
		}

		m_stream_profile_list = rs2_pipeline_profile_get_streams(m_pipeline_profile, &e);
		if (e)
		{
			xprintf("Failed to create stream profile list!\n");
			exit(EXIT_FAILURE);
		}

		m_stream_profile = (rs2_stream_profile*)rs2_get_stream_profile(m_stream_profile_list, 0, &e);
		if (e)
		{
			xprintf("Failed to create stream profile!\n");
			exit(EXIT_FAILURE);
		}

		rs2_get_stream_profile_data(m_stream_profile, &m_stream, &m_format, &m_index, &m_unique_id, &m_framerate, &e);
		if (e)
		{
			xprintf("Failed to get stream profile data!\n");
			exit(EXIT_FAILURE);
		}

		rs2_get_video_stream_resolution(m_stream_profile, &m_width, &m_height, &e);
		if (e)
		{
			xprintf("Failed to get video stream resolution data!\n");
			exit(EXIT_FAILURE);
		}

		int display_size = m_width * m_height;
		m_buffer_size = display_size * sizeof(uint16_t);

		for (int i = 0; i < m_numFrames; i++)
			m_buffers[i] = new uint16_t[display_size];

		xprintf("End of %s with no blow-ups: %d x %d, bufferSize: %d.\n", __FUNCTION__, m_width, m_height, m_buffer_size);
	}

	void TerminateRealSense() {
		// Release resources
		rs2_delete_pipeline_profile(m_pipeline_profile);
		rs2_delete_stream_profiles_list(m_stream_profile_list);
		rs2_delete_stream_profile(m_stream_profile);
		rs2_delete_config(m_config);
		rs2_delete_pipeline(m_pipeline);
		rs2_delete_device(m_dev);
		rs2_delete_device_list(m_device_list);
		rs2_delete_context(m_ctx);
	}

	void ShowFrameParams(rs2_frame* frame) {
		rs2_error* e{ nullptr };

		xprintf("Frame: %d x %d, %d\n", rs2_get_frame_width(frame, &e), rs2_get_frame_height(frame, &e), rs2_get_frame_data_size(frame, &e));
		check_error(e);
	}

	void Run(void) {
		rs2_error* e{ nullptr };

		//InitRealSense();

		while (IsRunning()) {
			// This call waits until a new composite_frame is available
			// composite_frame holds a set of frames. It is used to prevent frame drops
			// The returned object should be released with rs2_release_frame(...)
			rs2_frame* frames = rs2_pipeline_wait_for_frames(m_pipeline, RS2_DEFAULT_TIMEOUT, &e);
			check_error(e);

			// Returns the number of frames embedded within the composite frame
			int num_of_frames = rs2_embedded_frames_count(frames, &e);
			check_error(e);

			for (int i = 0; i < num_of_frames; ++i)
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

				//ShowFrameParams(frame);

				// Here is where the visualization code needs to go to render the depth info.
				// When I figure out how.
				memcpy(m_buffers[m_wIdx++ & (m_numFrames-1)], depth_frame_data, m_buffer_size);

				rs2_release_frame(frame);
			}

			rs2_release_frame(frames);
			//std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(10));
		}

		xprintf("%s terminating\n", __FUNCTION__);
		rs2_pipeline_stop(m_pipeline, &e);
		check_error(e);
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

	void Draw() {
		glEnable(GL_BLEND);
		GL_CHECK("glEnable(GL_BLEND) failed");

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		GL_CHECK("glBlendFunc() failed");

		if (m_wIdx > m_rIdx) {
			int index = (m_rIdx++) % m_numFrames;

			if (m_buffers[index]) {
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_width, m_height, 0, GL_RED, GL_UNSIGNED_SHORT, (void *)(m_buffers[index]));
				GL_CHECK("glTexSubImage2D() failed");
			}
		}

		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");

		glDisable(GL_BLEND);
		GL_CHECK("glDisable(GL_BLEND) failed");
	}

private:
	rs2_config* m_config{ nullptr };
	rs2_device_list* m_device_list{ nullptr };
	rs2_device* m_dev{ nullptr };
	rs2_context* m_ctx{ nullptr };
	rs2_pipeline* m_pipeline{ nullptr };
	rs2_pipeline_profile* m_pipeline_profile{ nullptr };
	rs2_stream_profile_list* m_stream_profile_list{ nullptr };
	rs2_stream_profile* m_stream_profile{ nullptr };
	rs2_stream m_stream{ RS2_STREAM_ANY };
	rs2_format m_format{ RS2_FORMAT_ANY };
	int m_dev_count{ 0 };
	int m_index{ -1 };
	int m_unique_id{ 0 };
	int m_framerate{ 0 };

	uint16_t m_one_meter{ 0 };
	uint16_t* m_buffers[m_numFrames]{ nullptr };
	int m_buffer_size{ 0 };
	int m_width{ 0 }; 
	int m_height{ 0 };
	uint64_t m_wIdx{ 0 };
	uint64_t m_rIdx{ 0 };
};

void ExampleXGL::BuildScene() {
	XGLDepthCam *depthCam;
	XGLDepthCloud *depthCloud;

	AddShape("shaders/flat", [&depthCloud]() { depthCloud = new XGLDepthCloud(1024, 768); return depthCloud; });
	depthCloud->attributes.diffuseColor = XGLColors::white;
	depthCloud->model = glm::scale(glm::mat4(), glm::vec3(4*10.24f, 4*7.68f, 1.0f));

	AddPreRenderFunction(depthCloud->invokeComputeShader);

	/*
	AddShape("shaders/zval_in_red", [&depthCam]() { depthCam = new XGLDepthCam(); return depthCam; });
	depthCam->attributes.diffuseColor = XGLColors::white;
	depthCam->model = glm::scale(glm::mat4(), glm::vec3(10.24f, 7.68f, 1.0f));

	depthCam->Start();
	*/
}
