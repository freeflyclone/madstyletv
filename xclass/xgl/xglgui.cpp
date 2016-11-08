#include "xglgui.h"

XGLGuiCanvas::XGLGuiCanvas(int w, int h) :
	XGLTexQuad(),
	width(w),
	height(h)
{
	xprintf("XGLGuiCanvas::XGLGuiCanvas()\n");
}

XGLGuiCanvas::~XGLGuiCanvas() {
	xprintf("XGLGuiCanvas::~XglGuiCanvas()\n");
}