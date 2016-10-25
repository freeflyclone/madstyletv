#include "xgl.h"

XGLPixelbuffer::XGLPixelbuffer(int w, int h) {

}

XGLPixelbuffer::~XGLPixelbuffer() {}

void XGLPixelbuffer::Render() {
	xprintf("XGLPixelbuffer::Render()\n");
}

void XGLSharedPBO::Render() {
	width = pHeader->width;
	height = pHeader->height;

	xprintf("XGLSharedPBO::Render()\n");

	XGLPixelbuffer::Render();
}