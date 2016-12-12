#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiWindow *gw;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 20, 20, 440, 60); return gw; });
	gw->attributes.diffuseColor = XGLColors::yellow;
	gw->SetPenPosition(10, 20);
	gw->RenderText("This window is pinned to the upper left corner. (the default)\nThis is a test, just to see if this works.\n", 16);

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 0, 0, 400, 80); return gw; });
	gw->attributes.diffuseColor = XGLColors::white;
	gw->SetPenPosition(10, 20);
	gw->RenderText("This window is pinned to the upper right corner,\nvia a reshape callback.\n\nText does not automatically wrap, it just gets clipped.", 16);
	gm->AddReshapeCallback([gw](int w, int h) {
		gw->model = glm::translate(glm::mat4(), glm::vec3(w - gw->width - 20, 20, 0.0));
	});

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 0, 0, 540, 80); return gw; });
	gw->attributes.diffuseColor = XGLColors::cyan;
	gw->SetPenPosition(10, 20);
	gw->RenderText("This window is pinned to the lower right corner, also via a callback.\n\n", 16);
	gw->RenderText("It's possible to change font size on the fly.\n", 20);
	gm->AddReshapeCallback([gw](int w, int h) {
		gw->model = glm::translate(glm::mat4(), glm::vec3(w - gw->width - 20, h - gw->height - 20, 0.0));
	});

	gm->AddChildShape("shaders/ortho", [&]() { gw = new XGLGuiWindow(this, "HorizontalSliderWindow", 20, 100, 360, 180); return gw; });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Horizontal Slider 1", XGLGuiSlider::Orientation::horizontal, 20, 20, 200, 16); });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Horizontal Slider 2", XGLGuiSlider::Orientation::horizontal, 20, 60, 200, 16); });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Horizontal Slider 3", XGLGuiSlider::Orientation::horizontal, 20, 100, 200, 16); });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Horizontal Slider 4", XGLGuiSlider::Orientation::horizontal, 20, 140, 200, 16); });

	return;
}
