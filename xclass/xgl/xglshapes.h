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
#include "AntTweakBar.h"

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
	XGLTexQuad(int width, int height);
	XGLTexQuad();
	virtual void Draw();
};

class XGLTransformer : public XGLShape {
public:
	XGLTransformer();
};

class XGLGuiCanvas : public XGLTexQuad {
public:
	typedef std::function<bool(XGLShape *, float x, float y, int f)> MouseFunc;

	XGLGuiCanvas(XGL *xgl, int w, int h);
	XGLGuiCanvas(XGL *xgl, int x, int y, int w, int h);
	XGLGuiCanvas(XGL *xgl);

	void SetXGL(XGL *xgl) { pxgl = xgl; }
	void SetFocus(bool enable) { hasFocus = enable; }
	bool HasFocus() { return hasFocus; }

	void SetHasMouse(bool enable) { hasMouse = enable; }
	bool HasMouse() { return hasMouse; }

	void SetMouseFunc(XGLGuiCanvas::MouseFunc);
	bool MouseEvent(float x, float y, int flags);

	void Reshape(int w, int h);

	void RenderText(std::wstring t);
	void Fill(GLubyte val);

	~XGLGuiCanvas();

	XGLGuiCanvas::MouseFunc mouseFunc;
	bool childEvent;
	int width, height;
	int xOrig, yOrig;
	int windowWidth, windowHeight;
	glm::mat4 orthoProjection;

private:
	GLubyte *buffer;
	bool hasFocus;
	bool hasMouse;

	// text rendering stuff
	int penX, penY;

	XGL *pxgl;
};

class XGLGuiCanvasWithReshape : public XGLGuiCanvas {
public:
	XGLGuiCanvasWithReshape(XGL *, int w, int h);

	void Reshape(int w, int h);

	int ww, wh, wx, wy;
};


class XGLAntTweakBar : public XGLShape {
public:
	XGLAntTweakBar(XGL *xgl);
	~XGLAntTweakBar();

	void Draw();
	void Reshape(int w, int h);
	void MouseMotion(int x, int y, int f);

	XGL *pxgl;
	int flags;
};


#endif // XGLSHAPES_H
