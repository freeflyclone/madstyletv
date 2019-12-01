// New concept: use additional XGLShape(s) to render visualizations of XGLFreeType structures
// while I'm developing my hybrid shader-based Freetype2 renderer.

#ifndef XGLFREETYPEUTILS_H
#define XGLFREETYPEUTILS_H

#include "XGLFreeType.h"

class XGLFreetypeGrid : public XGLShape {
public:
	XGLFreetypeGrid(XGL* pxgl);
	void Update(XGLVertexList vList, FT::BoundingBox bb);
	void Draw();

	void Move(int);

	XGL* pXgl;
	FT::BoundingBox bb;
	bool draw{ true };
	bool drawBorder{true};
	bool drawUpTo{ true };
	bool drawFromHere{ true };
	int idx{ 0 };
};

class XGLFreetypeProbe : public XGLShape {
public:
	XGLFreetypeProbe(XGL* pxgl);
	void Move(XGLVertex v, FT::BoundingBox bb);

	XGL* pXgl{ nullptr };
	XGLSphere *sphere{ nullptr }, *sphereX{ nullptr }, *sphereY{ nullptr };
};


class XGLFreetypeCrosshair : public XGLShape {
public:
	XGLFreetypeCrosshair(XGL* pxgl);
	void Update(XGLVertexList vList, FT::BoundingBox b);
	void Draw();

	void Move(int);

	XGL* pXgl{ nullptr };
	FT::BoundingBox bb;
	bool draw{ true };
	int idx{ 0 };
};

class XGLFreetypeNearest : public XGLShape {
public:
	XGLFreetypeNearest(XGL* pxgl);
	void Update(XGLVertexList vList);
	void Draw();

	XGL* pXgl{ nullptr };
	bool draw{ true };

	XGLVertexList nn;
};

#endif