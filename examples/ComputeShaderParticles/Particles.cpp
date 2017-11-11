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
