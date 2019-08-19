#ifndef XGLFREETYPEUTILS_H
#define XGLFREETYPEUTILS_H

#include "XGLFreeType.h"

class XGLFreetypeGrid : public XGLShape {
public:
	XGLFreetypeGrid(XGL* pxgl, XGLVertexList vList, FT::BoundingBox bb);
	void Draw();

	XGL* pXgl;
	bool drawGrid{ true };
};

// New concept: use additional XGLShape(s) to render visualizations of XGLFreeType structures
// while I'm developing my hybrid shader-based Freetype2 renderer.
class XGLFreetypeProbe : public XGLShape {
public:
	XGLFreetypeProbe(XGL* pxgl);
	void Move(XGLVertex v, FT::BoundingBox bb);

	XGL* pXgl{ nullptr };
	XGLCube *cube{ nullptr }, *cubeX{ nullptr }, *cubeY{ nullptr };
};



#endif