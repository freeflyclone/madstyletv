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
class GuiManager : public XGLGuiCanvas {
public:
	typedef std::function<void(int, int)> ReshapeCallback;
	typedef std::vector<ReshapeCallback> ReshapeCallbackList;

	GuiManager(XGL *xgl, bool addTexture = false) : XGLGuiCanvas(xgl), pxgl(xgl), padding(20) {
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

	void AddReshapeCallback(ReshapeCallback fn) {
		reshapeCallbacks.push_back(fn);
	}

	void Reshape(int w, int h) {
		for (ReshapeCallbackList::iterator rc = reshapeCallbacks.begin(); rc < reshapeCallbacks.end(); rc++)
			(*rc)(w, h);
	}

	XGL *pxgl;
	int padding;
	ReshapeCallbackList reshapeCallbacks;
};

void ExampleXGL::BuildGUI() {
	GuiManager *gm;
	XGLGuiCanvas *g;

	// Instantiate a GuiManager shape, which serves as GuiRoot() shape.
	// XGL::GuiResolve() requires a "place holder" as the root shape to
	// allow for recursively passing mouse events to the XGLGuiCanvas hierarchy.
	AddGuiShape("shaders/ortho", [&]() { gm = new GuiManager(this); return gm; });

	// The first XGLGuiCanvas in the stack, henceforth the "background canvas", 
	// is the "bottom-most" in Z stack order.
	// This shape specifies a ReshapeCallback to allow it to always exactly cover the window
	gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(this, 1, 1, false); SetName("GuiBackground");  return g; });
	g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.05 };
	gm->AddReshapeCallback([g](int w, int h) {
		g->width = w;
		g->height = h;
		g->Reshape(0, 0, w, h);
	});

	bool exampleTextWindow = true;
	if (exampleTextWindow) {
		XGLGuiCanvas *g2;
		gm->AddChildShape("shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 380, 120); return g2; });
		g2->model = glm::translate(glm::mat4(), glm::vec3(20, 20, 0));
		g2->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.8 };
		g2->SetPenPosition(10, 24);
		g2->RenderText("This box is pinned to the upper left corner\n\nThis is at the same GUI stack hierarchy level\nas the background canvas, and therefore\n should be \"under\" what gets created later.", 18);
	}

	bool exampleTextWindow2 = true;
	if (exampleTextWindow2) {
		XGLGuiCanvas *g2;
		gm->AddChildShape("shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 380, 120); return g2; });
		g2->model = glm::translate(glm::mat4(), glm::vec3(800, 20, 0));
		g2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.8 };
		gm->AddReshapeCallback([g2](int w, int h) {
			g2->model = glm::translate(glm::mat4(), glm::vec3(w - g2->width - 20, 20, 1.0));
		});
		g2->SetPenPosition(10, 24);
		g2->RenderText("This box is pinned to the upper right corner.\n\nIt is not currently possible to\nauto-wrap text, so clipping is used instead.", 18);
	}

	bool exampleHorizontalSlider = true;
	if (exampleHorizontalSlider) {
		XGLGuiCanvas *g2,*g3;

		// add the "track" for the horizontal slider.  We want it to hug the bottom, therefore
		// it's height is important to know. However it's width is dynamic according to window
		// size, so the initial value for width is irrelevant. The ReshapeCallback specifies
		// the desired layout behavior.
		gm->AddChildShape("shaders/ortho", [&]() { g2 = new XGLGuiCanvas(this, 1, 16); return g2; });
		g2->SetName("HorizontalSlider");
		g2->attributes.diffuseColor = { 1.0, 0.2, 0.2, 0.1 };
		gm->AddReshapeCallback([g2](int w, int h) {
			int padding = 20;
			g2->width = w - 2 * padding;
			g2->model = glm::translate(glm::mat4(), glm::vec3(padding, h - g2->height - padding, 0.0));
			g2->Reshape(0, 0, g2->width, g2->height);
		});
		g2->SetMouseFunc([this, g2](float x, float y, int flags){
			if (flags & 1) {
				XGLGuiCanvas *slider = (XGLGuiCanvas *)(g2->Children()[0]);
				// constrain mouse X coordinate to dimensions of track
				float xLimited = (x<0)?0:(x>(g2->width-slider->width))?(g2->width-slider->width):x;
				static float previousXlimited = 0.0;

				if (xLimited != previousXlimited) {
					slider->model = glm::translate(glm::mat4(), glm::vec3(xLimited, 0.0, 0.0));
					previousXlimited = xLimited;
				}
				mouseCaptured = g2;
				g2->SetHasMouse(true);
			}
			else {
				mouseCaptured = NULL;
				g2->SetHasMouse(false);
			}
			return true;
		});
		g2->AddChildShape("shaders/ortho", [&]() { g3 = new XGLGuiCanvas(this, 16, 16); return g3; });
		g3->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.5 };
	}

	// The final window in the GUI stack is a wrapper for AntTweakBar, for development
	// purposes.  Not sure if I'm going to keep this, but it seemed like a good idea
	// when I integrated it.  Unfortunately, it doesn't look professional enough
	// for end-product use, IMHO, and is lacking features that I want.
	bool enableAntTweakBar = false;
	if (enableAntTweakBar) {
		gm->AddChildShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });
	}

	return;
}
