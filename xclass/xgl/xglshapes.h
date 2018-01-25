/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XGLSHAPES_H
#define XGLSHAPES_H

#include "xglprimitives.h"
#include "xglbuffer.h"
#include "xglshader.h"
#include "xglmaterial.h"
#include "xphybody.h"

class XGLShape : public XObject , public XGLBuffer, public XGLMaterial, public XPhyBody {
public:
    typedef std::function<void(float)> AnimationFn;
	typedef std::function<void()> DrawFn;

    XGLShape();
    virtual ~XGLShape();

	virtual void Draw() {};

    void Animate(float clock);
	void SetAnimationFunction(AnimationFn);

    void Transform(glm::mat4 tm);
	void SetColor(XGLColor c);

	virtual void Render();
	virtual void Render(glm::mat4 model);

	void AddChild(XGLShape *s);

	XGLShape *Parent() { return parent; }
    XGLVertexList v;
    XGLIndexList idx;

	AnimationFn animationFunction;
	AnimationFn preRenderFunction;
	AnimationFn postRenderFunction;

	//glm::mat4 model;

	bool isVisible;
	XGLShape* parent;
};

// define a type for passing a lambda that creates an XGLShape as an argument
typedef std::function<XGLShape *()> XGLNewShapeLambda;

class XGLAxis : public XGLShape {
public:
	XGLAxis(float length = 5.0f, XGLColor color = { 1.0, 0.0, 0.0, 1.0 }, XGLVertex vertex = { 1.0, 0.0, 0.0 });
	void Draw();
};

class XYPlaneGrid : public XGLShape {
public:
	XYPlaneGrid(float size=100.0f, float step=10.0f);
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

	float radius; 

private:
    int nSegments;
	bool visualizeNormals;
};

class XGLHemiSphere : public XGLShape {
public:
	XGLHemiSphere(float r, int n);
	void Draw();

	float radius;

private:
	int nSegments;
	bool visualizeNormals;
};
class XGLCapsule : public XGLShape {
public:
	XGLCapsule(float, float, int);
	virtual ~XGLCapsule();
	virtual void Draw();

	float length, radius;

private:
	int nSegments;
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

class XGLSled : public XGLShape {
public:
	XGLSled(bool sa = false);

	void Draw();
	glm::mat4 GetFinalMatrix();
	void SampleInput(float yaw, float pitch, float roll);

private:
	bool showAxes;
};

class XGLPointCloud : public XGLShape {
public:
	XGLPointCloud(int nPoints = 1024, float radius = 5.0f, XGLColor color = { 1.0, 1.0, 1.0, 1.0 }, XGLVertex center = { 0.0, 0.0, 0.0 });
	void Draw();
	DrawFn drawFn;
};

#endif // XGLSHAPES_H
