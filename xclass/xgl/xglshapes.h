/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XGLSHAPES_H
#define XGLSHAPES_H

#include "xglobject.h"
#include "xglprimitives.h"
#include "xglbuffer.h"
#include "xglshader.h"
#include "xglmaterial.h"

class XGLShape : public XGLObject , public XGLBuffer, public XGLMaterial {
public:
    typedef std::function<void(XGLShape *, float)> AnimaFunk;

    XGLShape();
    virtual ~XGLShape();

	virtual void Draw() {};

    void Animate(float clock);
    void SetTheFunk(AnimaFunk);

    void Transform(glm::mat4 tm);
	void SetColor(XGLColor c);

	virtual void Render(float clock = 0.0f);
	virtual void Render(glm::mat4 model, float clock);

    XGLVertexList v;
    XGLIndexList idx;

    AnimaFunk funk;
	AnimaFunk preRenderFunction;
	AnimaFunk postRenderFunction;

	glm::mat4 model;
};

class XYPlaneGrid : public XGLShape {
public:
	XYPlaneGrid();
	void Draw();
};

class XGLTriangle : public XGLShape {
public:
    XGLTriangle();
    void Draw();
};

class XGLCube : public XGLShape {
public:
	XGLCube();
    void Draw();
};
class XGLSphere : public XGLShape {
public:
    XGLSphere(float r, int n);
    void Draw();

private:
    int nSegments;
    float radius;
	bool visualizeNormals;
};

class XGLCapsule : public XGLShape {
public:
	XGLCapsule(float, float, int);
	virtual ~XGLCapsule();
	virtual void Draw();

private:
	int nSegments;
	float length, radius;
};

class XGLSphere2 : public XGLShape {
public:
	XGLSphere2(float r, int n);
	void Draw();

private:
	int nSegments;
	float radius;
	bool visualizeNormals;
	XGLIndexList idxEnd1;
	XGLIndexList idxEnd2;
};

class XGLIcoSphere : public XGLShape {
public:
	XGLIcoSphere();
	void Draw();
};

class XGLTorus : public XGLShape {
public:
    XGLTorus(float rMaj, float rMin, int nMaj, int nMin);
    void Draw();

private:
    int nSegmentsMajor, nSegmentsMinor;
    float radiusMajor, radiusMinor;
    GLsizei nTorusIndices,nTotalIndices;
    bool visualizeNormals;
};

class XGLTextureAtlas : public XGLShape {
public:
	XGLTextureAtlas();
	virtual void Draw();
private:
	float gridCellWidth, gridCellHeight;
	int gridXsize, gridYsize;
};

class XGLTexQuad : public XGLShape{
public:
	XGLTexQuad(std::string fileName);
	XGLTexQuad(int width, int height, int channels, GLubyte *img, bool flipColors = false);
	XGLTexQuad(int width, int height, int channels);
	XGLTexQuad();
	virtual void Draw();
};

class XGLTransformer : public XGLShape {
public:
	XGLTransformer();
};

class XGLGuiCanvas : public XGLTexQuad {
public:
	typedef std::function<void(XGLGuiCanvas *, float x, float y, int f)> MouseFunc;

	XGLGuiCanvas(int w, int h);

	void SetFocus(bool enable) { hasFocus = enable; }
	bool HasFocus() { return hasFocus; }

	void SetHasMouse(bool enable) { hasMouse = enable; }
	bool HasMouse() { return hasMouse; }

	void SetMouseFunc(XGLGuiCanvas::MouseFunc);
	void MouseEvent(float x, float y, int flags);

	void RenderText(std::wstring t);

	~XGLGuiCanvas();

	XGLGuiCanvas::MouseFunc mouseFunc;

private:
	int width, height;
	GLubyte *buffer;
	bool hasFocus;
	bool hasMouse;

	// text rendering stuff
	int penX, penY;
};

#endif // XGLSHAPES_H
