#include "ExampleXGL.h"


void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiCanvas *g;

	// Instantiate a GuiManager shape, which serves as GuiRoot() shape.
	// XGL::GuiResolve() requires a "place holder" as the root shape to
	// allow for recursively passing mouse events to the XGLGuiCanvas hierarchy.
	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	// The first XGLGuiCanvas in the stack, henceforth the "background canvas", 
	// is the "bottom-most" in Z stack order.
	// This shape specifies a ReshapeCallback to allow it to always exactly cover the window
	gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(this, 1, 1, false); SetName("GuiBackground");  return g; });
	g->attributes.ambientColor = { 1.0, 1.0, 1.0, 0.05 };
	gm->AddReshapeCallback([g](int w, int h) {
		g->width = w;
		g->height = h;
		g->Reshape(0, 0, w, h);
	});

	bool exampleTextWindow = true;
	if (exampleTextWindow) {
		XGLGuiCanvas *g2;
		gm->AddChildShape("shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 304, 24); return g2; });
		g2->model = glm::translate(glm::mat4(), glm::vec3(20, 20, 0));
		g2->attributes.diffuseColor = yellow;
		g2->attributes.ambientColor = { 0.0, 0.0, 0.0, 0.5 };
		g2->SetPenPosition(4, 17);
		g2->RenderText("This box is pinned to the upper left corner", 16);
	}

	bool exampleTextWindow2 = true;
	if (exampleTextWindow2) {
		XGLGuiCanvas *g2;
		gm->AddChildShape("shaders/ortho-tex", [&]() { g2 = new XGLGuiCanvas(this, 324, 84); return g2; });
		g2->model = glm::translate(glm::mat4(), glm::vec3(800, 20, 0));
		g2->attributes.diffuseColor = white;
		g2->attributes.ambientColor = { 0.0, 0.0, 0.0, 0.5 };
		gm->AddReshapeCallback([g2](int w, int h) {
			g2->model = glm::translate(glm::mat4(), glm::vec3(w - g2->width - 20, 20, 1.0));
		});
		g2->SetPenPosition(10, 24);
		g2->RenderText("This box is pinned to the upper right corner.\n\nIt is not currently possible to\nauto-wrap text, so clipping is used instead.", 16);
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
		g2->attributes.ambientColor = { 1.0, 0.2, 0.2, 0.1 };
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
		g3->attributes.ambientColor = { 1.0, 1.0, 0.0, 0.5 };
	}

	return;
}
