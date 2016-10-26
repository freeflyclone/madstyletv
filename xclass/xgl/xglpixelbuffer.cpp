#include "xgl.h"

XGLPixelbuffer::XGLPixelbuffer(int w, int h) : width(w), height(h){
}

XGLPixelbuffer::~XGLPixelbuffer() {
}

void XGLPixelbuffer::Render(unsigned char *b) {
}

void XGLSharedPBO::Render() {
	width = pHeader->width;
	height = pHeader->height;
}