#ifndef XGLGUI_H
#define XGLGUI_H

#include "xgl.h"

class XGLGuiCanvas : public XGLTexQuad {
public:
	XGLGuiCanvas(int w, int h);

	~XGLGuiCanvas();
private:
	int width, height;
	bool isVisible;
};
#endif