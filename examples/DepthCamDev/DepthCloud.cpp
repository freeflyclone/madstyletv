#include "DepthCloud.h"

XGLDepthCloud::XGLDepthCloud(int w, int h) : XGLPointCloud(), m_width(w), m_height(h) {
	SetName("XGLDepthCloud");

	GenUShortSSBO(m_width, m_height);

	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxInvocations);
	xprintf("Max Invocations: %d\n", maxInvocations);

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			float xCoord = (float)x / (float)m_width;
			float yCoord = (float)y / (float)m_height;
			XGLVertexAttributes vrtx;

			vrtx.v = { xCoord, yCoord, 0.0 };
			vrtx.t = { xCoord, yCoord };
			vrtx.c = XGLColors::white;
			vrtx.n = { 0.0, 0.0, 1.0 };

			v.push_back(vrtx);
		}
	}

	// create a compute shader object for this XGLDepthCloud
	m_computeShader = new XGLShader("shaders/depth-cloud");
	m_computeShader->CompileCompute(pathToAssets + "/shaders/depth-cloud");

	// Need to call AddPreRenderFunction() in XGL derived class (ExampleXGL::BuildScene) to add this lambda function to the its preRenderFunctions list
	invokeComputeShader = [this](float clock) {
		glUseProgram(m_computeShader->programId);

		// map our VBO as SSBO, so compute shader can diddle our VBO objects
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo);
		GL_CHECK("Eh, Something failed");

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssboId);
		GL_CHECK("Eh, Something failed");

		// fire the compute shader
		glDispatchCompute(maxInvocations, 1, 1);

		// wait until the compute shader has completed before rendering it's results
		glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
		GL_CHECK("Dispatch compute shader");
	};
}

void XGLDepthCloud::GenUShortSSBO(const int width, const int height) {
	int requiredSize = width * height * sizeof(uint16_t);

	glGenBuffers(1, &m_ssboId);
	GL_CHECK("Eh, something failed");

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboId);
	GL_CHECK("Eh, something failed");

	glBufferData(GL_SHADER_STORAGE_BUFFER, requiredSize, nullptr, GL_DYNAMIC_DRAW);
	GL_CHECK("Eh, something failed");

	return;
}


void XGLDepthCloud::Draw() {
	if (v.size()) {
		glPointSize(4.0f);

		glDrawArrays(GL_POINTS, 0, (GLuint)v.size());
		GL_CHECK("glDrawArrays() failed");
	}
}

