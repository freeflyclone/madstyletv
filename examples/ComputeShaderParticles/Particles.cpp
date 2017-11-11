#include "Particles.h"

GLuint createNoiseTexture4f3D(int w = 16, int h = 16, int d = 16, GLint internalFormat = GL_RGBA8_SNORM)
{
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

	GLuint tex;
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

XGLParticleSystem::XGLParticleSystem(int n) : XGLPointCloud(0) {
	SetName("XGLParticleSystem");

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 10.0);

	VertexAttributes vrtx;
	for (int i = 0; i < n; i++) {
		vrtx.pos.x = dis(gen);
		vrtx.pos.y = dis(gen);
		vrtx.pos.z = dis(gen);
		vrtx.pos.w = 1.0f;
		vrtx.color = { 1.0, 1.0, 1.0, 1.0 };
		verts.push_back(vrtx);
	}

	// v.size() must be non-zero else XGLShape::Render(glm::mat4) won't do all the things
	v.push_back({});

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

	// unbind now that we're done.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	GL_CHECK("Oops, something bad happened");
}

void XGLParticleSystem::Draw() {
	glPointSize(4.0f);

	// have to bind our custom VAO here, else we get XGLBuffer::vao, which is NOT what we want
	glBindVertexArray(vao);
	GL_CHECK("glBindVertexArray() failed");

	// need our custom VBO bound as well
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	GL_CHECK("glBindBuffer() failed");

	// draw custom VBO per custom VAO layout
	glDrawArrays(GL_POINTS, 0, (GLuint)verts.size());
	GL_CHECK("glDrawArrays() failed");
}