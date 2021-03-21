/**
	XGLDepthCloud a GLSL compute shader based depth cloud system.

	An example of sharing a VBO with a compute shader, so as to affect vertex positions via
	the compute shader code.

	This is tailored for use with the Intel RealSense L515 LIDAR depth camera. The camera
	is capable of 1024 x 768 array of depth data at 16bit resolution.
*/

#ifndef DEPTHCLOUD_H
#define DEPTHCLOUD_H

#include "ExampleXGL.h"

class XGLDepthCloud : public XGLPointCloud {
public:
	XGLDepthCloud(int w = 0, int h = 0);
	virtual void Draw();

	AnimationFn invokeComputeShader;

private:
	XGLShader *m_computeShader{ nullptr };
	int m_width, m_height;
	int maxInvocations{ 0 };
};



#endif
