#include "tripod.h"

XGLTripod::XGLTripod() {
	SetName("XGLTripod");

	v.push_back({{0,0,0},{},{0,0,1}, XGLColors::red});
	v.push_back({{1,0,0},{},{0,0,1}, XGLColors::red});

	v.push_back({{0,0,0},{},{0,0,1}, XGLColors::green});
	v.push_back({{0,1,0},{},{0,0,1}, XGLColors::green});

	v.push_back({{0,0,0},{},{0,0,1}, XGLColors::blue});
	v.push_back({{0,0,1},{},{0,0,1}, XGLColors::blue});

	for (int i=0; i<v.size(); i++)
		idx.push_back(i);
}

void XGLTripod::Draw() {
	glDrawElements(GL_LINES, (GLsizei)(idx.size()), XGLIndexType, 0);
	GL_CHECK("glDrawElements() failed");
    return;
}
