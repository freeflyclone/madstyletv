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

class XGLDepthCloud : public XGLShape {
public:
	/*
	struct VertexAttributes {
		glm::vec4 pos;
		glm::vec4 norm;
		glm::vec4 color;
	};
	*/
	typedef XGLVertexAttributes VertexAttributes;
	typedef std::vector<VertexAttributes> VertexList;

	XGLDepthCloud(int n = 0);
	virtual void Draw();

	AnimationFn invokeComputeShader;

private:
	VertexList verts;
	GLuint vbo{ 0 }, vao{ 0 }, tex{ 0 };
	XGLShader *computeShader{ nullptr };
	int numParticles{ 0 };
	int maxInvocations{ 0 };
	int sx{ 0 }, sy{ 0 }, sz{ 0 };
	int cx{ 0 }, cy{ 0 }, cz{ 0 };
};



#endif
