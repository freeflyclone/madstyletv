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

// define a type for passing a lambda that creates an XGLShape as an argument
typedef std::function<XGLShape *()> XGLNewShapeLambda;

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
	// these constructors build a quad @ -1,-1 to 1,1
	XGLTexQuad();
	XGLTexQuad(std::string fileName);
	XGLTexQuad(int width, int height, int channels, GLubyte *img, bool flipColors = false);
	XGLTexQuad(int width, int height, int channels);

	virtual ~XGLTexQuad(){};

	// special case: build a quad @ 0,0 to width,height in the vertices
	// (useful for screenspace (GUI) quads)
	XGLTexQuad(int width, int height);

	void Reshape(int left, int top, int width, int height);

	virtual void Draw();
};

class XGLTransformer : public XGLShape {
public:
	XGLTransformer();
};

class XGLGuiCanvas : public XGLTexQuad {
public:
	typedef std::function<bool(float x, float y, int f)> MouseFunc;
	typedef std::function<void(float x, float y, int f)> MouseEventListener;
	typedef std::vector<MouseEventListener> MouseEventListeners;

	// uses default XGLTexQuad() constructor
	XGLGuiCanvas(XGL *xgl);

	// uses special XGLTexQuad(int w, int h) constructor
	XGLGuiCanvas(XGL *xgl, int w, int h, bool addTexture = true);

	void AddChildShape(std::string shaderName, XGLNewShapeLambda fn);

	void SetXGL(XGL *xgl) { pxgl = xgl; }
	void SetFocus(bool enable) { hasFocus = enable; }
	bool HasFocus() { return hasFocus; }

	void SetHasMouse(bool enable) { hasMouse = enable; }
	bool HasMouse() { return hasMouse; }

	void SetMouseFunc(XGLGuiCanvas::MouseFunc);
	bool MouseEvent(float x, float y, int flags);

	void AddMouseEventListener(XGLGuiCanvas::MouseEventListener);

	void RenderText(std::wstring t, int pixelSize=64);
	void RenderText(std::string t, int pixelSize = 64);
	void SetPenPosition(int x, int y) { penX = x; penY = y; }
	void Fill(GLubyte val);

	virtual ~XGLGuiCanvas();

	XGLGuiCanvas::MouseFunc mouseFunc;
	bool childEvent;
	int width, height;

private:
	GLubyte *buffer;
	bool hasFocus;
	bool hasMouse;

	// text rendering stuff
	int penX, penY;

	XGL *pxgl;
	MouseEventListeners mouseEventListeners;
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
	TwBar *bar;
};


#endif // XGLSHAPES_H
