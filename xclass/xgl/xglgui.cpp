#include "xglgui.h"

XGLGuiCanvas::XGLGuiCanvas(int w, int h) :
	XGLTexQuad(),
	width(w),
	height(h)
{
	attributes.diffuseColor = { 0.0, 0.0, 0.0, 0.5 };
	model = glm::scale(glm::mat4(), glm::vec3(0.95, 0.89, 1.0));
	xprintf("XGLGuiCanvas::XGLGuiCanvas()\n");
}

XGLGuiCanvas::~XGLGuiCanvas() {
	xprintf("XGLGuiCanvas::~XglGuiCanvas()\n");
}