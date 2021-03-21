#include "DepthCloud.h"

XGLDepthCloud::XGLDepthCloud(int n) : numParticles(n) {
	SetName("XGLDepthCloud");

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cx);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &cy);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &cz);
	xprintf("Max compute work group count = %d, %d, %d\n", cx, cy, cz);

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &sx);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &sy);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &sz);
	xprintf("Max compute work group size  = %d, %d, %d\n", sx, sy, sz);

	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxInvocations);
	xprintf("Max Invocations: %d\n", maxInvocations);

	for (int y = 0; y < 768; y++) {
		for (int x = 0; x < 1024; x++) {
			float xCoord = (float)x / 1024.0f;
			float yCoord = (float)y / 768.0f;
			VertexAttributes vrtx;

			vrtx.v = { xCoord, yCoord, 0.1 };
			vrtx.c = XGLColors::white;
			vrtx.n = { 0.0, 0.0, 1.0 };

			verts.push_back(vrtx);
		}
	}

	// v.size() must be non-zero else XGLShape::Render(glm::mat4) won't do all the things,
	// specifically, it won't setup the model matrix, call XGLBuffer::Bind() or XGLMaterial::Bind()
	// which are all necessary for rendering in the XGL framework.  So adding one point to
	// the default XGLVertexAttributes buffer for this shape  gets us what we need.
	// It ignored because we're about to override it with what comes next.
	v.push_back({});

	// Override the default XGLVertexAttributes list for this shape.

	// Using a custom vertex array object allows for non-XGLVertexAttributes standard VBO layout
	glGenVertexArrays(1, &vao);
	GL_CHECK("glGenBuffers() failed");
	glBindVertexArray(vao);
	GL_CHECK("glBindVertexArray() failed");

	// custom VBO for now - (possibly reuse XGLBuffer::vbo?)
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// define the VBO layout we're using for this object
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), 0);			// pos
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)16);	// norm
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)32);	// color

	// custom VAO, so enable what we want enabled.
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	// 'size' (2nd) arg is in bytes!!
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(VertexAttributes), verts.data(), GL_DYNAMIC_DRAW);

	// unbind now that we're done. Perhaps (probably?) superfluous.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	GL_CHECK("Oops, something bad happened");

	// create a compute shader object for this XGLParticleSystem
	computeShader = new XGLShader("shaders/depth-cloud");
	computeShader->CompileCompute(pathToAssets + "/shaders/depth-cloud");

	// Need to call AddPreRenderFunction() in XGL derived class (ExampleXGL::BuildScene) to add this lambda function to the its preRenderFunctions list
	invokeComputeShader = [this](float clock) {
		glUseProgram(computeShader->programId);

		// This is the magic: nothing special about a VBO, it's just a buffer.
		// So is an SSBO.  So just bind the VBO as an SSBO and now the compute shader
		// can access it.  Of course the compute shader and vertex shader have to agree
		// on the layout of the buffer, else mayhem ensues.
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo);

		// fire the compute shader
		glDispatchCompute(1024, 1, 1);

		// wait until the compute shader has completed before rendering it's results
		//glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
		//GL_CHECK("Dispatch compute shader");
	};
}

void XGLDepthCloud::Draw() {
	
	if (verts.size()) {
		glDisable(GL_BLEND);
		glPointSize(4.0f);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, tex);
		GL_CHECK("failed");

		// have to bind our custom VAO here, else we get XGLBuffer::vao, which is NOT what we want
		glBindVertexArray(vao);
		//GL_CHECK("glBindVertexArray() failed");

		// need our custom VBO bound as well
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		//GL_CHECK("glBindBuffer() failed");

		// draw custom VBO per custom VAO layout
		glDrawArrays(GL_POINTS, 0, (GLuint)verts.size());
		//GL_CHECK("glDrawArrays() failed");
	}
}

