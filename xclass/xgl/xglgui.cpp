#include "xglgui.h"

XGLGuiCanvas::XGLGuiCanvas(int w, int h) :
	XGLTexQuad(),
	width(w),
	height(h),
	isVisible(true)
{
	xprintf("XGLGuiCanvas::XGLGuiCanvas()\n");
}

XGLGuiCanvas::~XGLGuiCanvas() {
	xprintf("XGLGuiCanvas::~XglGuiCanvas()\n");
}