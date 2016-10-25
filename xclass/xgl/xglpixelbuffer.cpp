#include "xgl.h"

XGLPixelbuffer::XGLPixelbuffer(int w, int h) : width(w), height(h){}

XGLPixelbuffer::~XGLPixelbuffer() {}

void XGLPixelbuffer::Render(unsigned char *b) {
	glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, b);
}

void XGLSharedPBO::Render() {
	width = pHeader->width;
	height = pHeader->height;

	XGLPixelbuffer::Render(mappedBuffer);
}