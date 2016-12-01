#include "ExampleXGL.h"

/*
** GuiManager class
**
**	Adds a ReshapeCallback layer to XGLGuiCanvas items, and serves as the GuiRoot() shape.
**	The primary function of GuiManager is to manage the layout of child XGLGuiCanvas
**	items by allowing XGLGuiCanvas items that care about window sizing events to
**	get notification via callback functions.  It is envisioned that this will apply
**	mostly XGLGuiCanvas items that are intended to hug the right and/or bottom edges
**  of the main window.
**
**	Additionally, an XInput::XInputKeyFunc is added to allow dynamic toggling of
**  the GUI layer.  The GUI layer is maintained by XGL as a separate list of
**	XGLShape items.  This allows leveraging the existing hierarchy mechanism
**	already established for XGLShapes.
**
**  The window resizing functionality is provided by XGLProjector, since it needs
**	to know the window dimensions for the glViewport() call and was the first
**  class to handle OS-specific window sizing events.  GuiManager specializes
**	that mechanism to provide context for XGLGuiCanvas items that care about window
**	size events, as not all XGLGuiCanvas items need to care.
**
**  The implementation allows for XGLGuiCanvas items to specify lambda functions
**  as needed for the ReshapeCallback functions, which allows a fine granularity
**	in functional specificity without excessive sub-classing.
*/
class GuiManager : public XGLTransformer {
public:
	typedef std::function<void(XGLGuiCanvas *, int, int)> ReshapeCallback;
	typedef std::pair<XGLGuiCanvas *, ReshapeCallback> ReshapePair;
	typedef std::vector<ReshapePair> ReshapeCallbackList;

	GuiManager(XGL *xgl, bool addTexture = false) : pxgl(xgl), padding(20) {
		SetName("GuiManager");

		XInput::XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
			const bool isDown = (flags & 0x8000) == 0;
			const bool isRepeat = (flags & 0x4000) != 0;

			if (isDown && pxgl->GuiIsActive())
				pxgl->RenderGui(false);
			else if (isDown)
				pxgl->RenderGui(true);
		};

		pxgl->AddKeyFunc('`', PresentGuiCanvas);
		pxgl->AddKeyFunc('~', PresentGuiCanvas);

		xgl->projector.AddReshapeCallback(std::bind(&GuiManager::Reshape, this, _1, _2));
	}

	void AddReshapeCallback(XGLGuiCanvas *s, ReshapeCallback fn) {
		reshapeCallbacks.push_back(ReshapePair(s, fn));
	}

	void Reshape(int w, int h) {
		for (rc = reshapeCallbacks.begin(); rc < reshapeCallbacks.end(); rc++)
			(rc->second)(rc->first, w, h);
	}

	XGL *pxgl;
	int padding;
	ReshapeCallbackList reshapeCallbacks;
	ReshapeCallbackList::iterator rc;
};

void ExampleXGL::BuildGUI() {
	GuiManager *gm;
	XGLGuiCanvas *g;

	// Instantiate a GuiManager shape, which serves as GuiRoot() shape.
	// XGL::GuiResolve() requires a "place holder" as the root shape to
	// allow for recursively passing mouse events to the XGLGuiCanvas hierarchy.
	AddGuiShape("shaders/ortho", [&]() { gm = new GuiManager(this); return gm; });

	// The first XGLGuiCanvas in the stack, henceforth the "background canvas", is the "bottom-most" 
	// in Z stack order and traps mouse events so that they don't leak into the 
	// 3D world mouse event handling while the GUI layer is being presented.
	// This shape specifies a ReshapeCallback to allow it to always exactly cover the window
	AddGuiShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(this, 1, 1, false); return g; });
	g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.05 };
	g->SetMouseFunc([&](XGLShape *s, float x, float y, int flags){
		if (flags & 1)
			mouseCaptured = (XGLGuiCanvas *)s;
		else
			mouseCaptured = NULL;
		return true;
	});
	gm->AddReshapeCallback(g, [](XGLGuiCanvas *gc, int w, int h) {
		gc->width = w;
		gc->height = h;
		gc->Reshape(0, 0, w, h);
	});

	// All subsequent XGLGuiCanvas items should be children of the background canvas

	bool exampleTextWindow = true;
	if (exampleTextWindow) {
		XGLGuiCanvas *g2;
		g->AddChildShape("shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 360, 640); return g2; });
		g2->model = glm::translate(glm::mat4(), glm::vec3(800, 20, 0));
		g2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
		g2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags){
			xprintf("In %s(%0.0f,%0.0f)\n", s->name.c_str(), x, y);
			if (flags & 1)
				mouseCaptured = (XGLGuiCanvas *)s;
			else
				mouseCaptured = NULL;
			return true;
		});
		gm->AddReshapeCallback(g2, [](XGLGuiCanvas *gc, int w, int h) {
			gc->model = glm::translate(glm::mat4(), glm::vec3(w - gc->width - 20, 20, 1.0));
		});
		g2->SetPenPosition(10, 24);
		g2->RenderText("This is a test.\nIt should be possible to\nauto-wrap text, to avoid\nvisual artifacts for long lines.\n", 18);
	}

	bool exampleHorizontalSlider = true;
	if (exampleHorizontalSlider) {
		XGLGuiCanvas *g2,*g3;

		// add the "track" for the horizontal slider.  We want it to hug the bottom, therefore
		// it's height is important to know. However it's width is dynamic according to window
		// size, so the initial value for width is irrelevant. The ReshapeCallback specifies
		// the desired layout behavior.
		g->AddChildShape("shaders/ortho", [&]() { g2 = new XGLGuiCanvas(this, 1, 16); return g2; });
		g2->attributes.diffuseColor = { 1.0, 0.2, 0.2, 0.1 };
		gm->AddReshapeCallback(g2, [](XGLGuiCanvas *gc, int w, int h) {
			int padding = 20;
			gc->width = w - 2 * padding;
			gc->model = glm::translate(glm::mat4(), glm::vec3(padding, h - gc->height - padding, 0.0));
			gc->Reshape(0, 0, gc->width, gc->height);
		});
		g2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags){
			XGLGuiCanvas *gc = (XGLGuiCanvas *)s;
			if (flags & 1) {
				XGLGuiCanvas *slider = (XGLGuiCanvas *)(s->Children()[0]);
				// constrain mouse X coordinate to dimensions of track
				float xLimited = (x<0)?0:(x>(gc->width-slider->width))?(gc->width-slider->width):x;
				// scale the value to a percentage
				float xScaled = xLimited / (gc->width - slider->width) * 100.0f;
				// only report when the value has actually changed
				static float previousXscaled = 0.0;

				if (xScaled != previousXscaled) {
					slider->model = glm::translate(glm::mat4(), glm::vec3(xLimited, 0.0, 0.0));
					xprintf("Slider: %0.3f\n", xScaled);
					previousXscaled = xScaled;
				}
				mouseCaptured = (XGLGuiCanvas *)s;
			}
			else
				mouseCaptured = NULL;
			return true;
		});

		g2->AddChildShape("shaders/ortho", [&]() { g3 = new XGLGuiCanvas(this, 16, 16); return g3; });
		g3->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.5 };
	}

	// The final window in the GUI stack is a wrapper for AntTweakBar, for development
	// purposes.  Not sure if I'm going to keep this, but it seemed like a good idea
	// when I integrated it.  Unfortunately, it doesn't look professional enough
	// for end-product use, IMHO, and is lacking features that I want.
	bool enableAntTweakBar = true;
	if (enableAntTweakBar) {
		g->AddChildShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });
	}

	return;
}
