#include "Particles.h"

XGLParticleSystem::XGLParticleSystem(int n) : numParticles(n) {
	SetName("XGLParticleSystem");

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

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> posDis(-5.0, 5.0);
	std::uniform_real_distribution<> velDis(-0.05, 0.05);
	std::uniform_real_distribution<> colorDis(0.0, 1.0);

	for (int i = 0; i < numParticles; i++) {
		VertexAttributes vrtx;

		vrtx.pos = { posDis(gen), posDis(gen), posDis(gen), 1.0 };
		vrtx.color = { colorDis(gen), colorDis(gen), colorDis(gen), 1.0 };
		//vrtx.vel = { velDis(gen), velDis(gen), velDis(gen), 0.0 };

		verts.push_back(vrtx);
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
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)16);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)32);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)48);

	// custom VAO, so enable what we want enabled.
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	// 'size' (2nd) arg is in bytes!!
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(VertexAttributes), verts.data(), GL_DYNAMIC_DRAW);

	// unbind now that we're done. Perhaps (probably?) superfluous.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	CreateNoiseTexture4f3D();

	GL_CHECK("Oops, something bad happened");

	// create a compute shader object for this XGLParticleSystem
	computeShader = new XGLShader("shaders/particle-system");
	computeShader->CompileCompute(pathToAssets + "/shaders/particle-system");

	// Need to call AddPreRenderFunction() in XGL derived class (ExampleXGL::BuildScene) to add this lambda function to the its preRenderFunctions list
	invokeComputeShader = [this](float clock) {
		glUseProgram(computeShader->programId);

		// This is the magic: nothing special about a VBO, it's just a buffer.
		// So is an SSBO.  So just bind the VBO as an SSBO and now the compute shader
		// can access it.  Of course the compute shader and vertex shader have to agree
		// on the layout of the buffer, else mayhem ensues.
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo);

		// fire the compute shader
		glDispatchCompute(1000, 100, 1);

		// wait until the compute shader has completed before rendering it's results
		//glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
		//GL_CHECK("Dispatch compute shader");
	};
}

void XGLParticleSystem::Draw() {
	//glPointSize(2.0f);

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

GLuint XGLParticleSystem::CreateNoiseTexture4f3D(int w, int h, int d, GLint internalFormat) {
	uint8_t *data = new uint8_t[w*h*d * 4];
	uint8_t *ptr = data;
	for (int z = 0; z<d; z++) {
		for (int y = 0; y<h; y++) {
			for (int x = 0; x<w; x++) {
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
			}
		}
	}

	glActiveTexture(GL_TEXTURE0);

	glGenTextures(1, &tex);
	GL_CHECK("glGenTextues() failed");
	glBindTexture(GL_TEXTURE_3D, tex);
	GL_CHECK("failed");

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	GL_CHECK("failed");

	//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, w, h, d, 0, GL_RGBA, GL_BYTE, data);
	GL_CHECK("failed");

	delete[] data;
	return tex;
}
