#include "xglshapes.h"
#include "glm.hpp"
#include "matrix_transform.hpp"
#include "type_ptr.hpp"

/*
** XGLGuiCanvas class
**
**  A base class for all GUI related 2D shapes, derived from XGLTexQuad.
**  
*/
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

	void RenderText(std::wstring t, int pixelSize = 64);
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

/*
** XGLGuiManager class
**
**	Adds a ReshapeCallback layer to XGLGuiCanvas items, and serves as the XGL::GuiRoot() shape.
**	This allows XGLGuiCanvas items interested in window sizing events to get notification via 
**  callback functions.  It is envisioned that this will apply mostly XGLGuiCanvas items 
**  that are intended to hug the right and/or bottom edges of the main window.
**
**  The ReshapeCallback function receives the width and height of the main window.
*/
class XGLGuiManager : public XGLGuiCanvas {
public:
	typedef std::function<void(int, int)> ReshapeCallback;
	typedef std::vector<ReshapeCallback> ReshapeCallbackList;

	XGLGuiManager(XGL *xgl, bool addTexture = false);
	void AddReshapeCallback(ReshapeCallback fn);

	XGL *pxgl;
	int padding;
	ReshapeCallbackList reshapeCallbacks;
};

/*
** XGLGuiWindow class
**
**  A basic "widget container", for visual grouping of various XGLGuiCanvas derived
**  items, like sliders, text fields, check boxes, etc.
*/
class XGLGuiWindow : public XGLGuiCanvas {
public:
	XGLGuiWindow(XGL *xgl, std::string name, int x, int y, int w, int h);
};

/*
** XGLGuiSlider: define the essence of a GUI slider control, that can be either vertical or horizontal.
**
** Much of the layout is defaulted to arbitrary choices according to my personal preferences.  My
** preferences have been influenced by others work. I didn't invent anything here. I'm just emulating
** what I've seen in ALL the major desktop OS GUI frameworks.
**
** The goal here is the minimum amount of code necessary to provide visibly acceptable layout with
** unsurprising event response.
*/
class XGLGuiSlider : public XGLGuiCanvas {
public:
	enum Orientation {
		vertical,
		horizontal
	};

	XGLGuiSlider(XGL *xgl, std::string name, Orientation o, int x, int y, int w, int h);
	void MeasureFontMetrics(std::string name);
	void AdjustForOrientation(Orientation orientation, int x, int y, int w, int h);

private:
	XGLGuiCanvas *groove, *thumb, *label;
	int fontHeight, baselineHeight, labelPadding, labelWidth, labelHeight;
	int grooveWidth, grooveHeight, grooveOffset, thumbSize, thumbX, thumbY, labelX, labelY;
	glm::mat4 labelOffset;
	Orientation orientation;
	static const int pixelSize = 12;
};

