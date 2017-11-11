#include "ExampleXGL.h"
#include <filesystem>

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiWindow *gw;
	XGLGuiSlider *slider;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });
	gm->AddChildShape("shaders/ortho", [&]() { gw = new XGLGuiWindow(this, "SliderWindow", 20, 20, 88, 500); return gw; });
	gw->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(this, "Roll Rate", XGLGuiSlider::Orientation::vertical, 36, 20, 16, 400); return slider; });

	gm->AddChildShape("shaders/ortho", [&]() { gw = new XGLGuiWindow(this, "HorizontalSliderWindow", 0, 20, 360, 180); return gw; });
	gw->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(this, "Horizontal Slider 1", XGLGuiSlider::Orientation::horizontal, 20, 20, 200, 16); return slider; });
	gw->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(this, "Horizontal Slider 2", XGLGuiSlider::Orientation::horizontal, 20, 60, 200, 16); return slider; });
	gw->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(this, "Horizontal Slider 3", XGLGuiSlider::Orientation::horizontal, 20, 100, 200, 16); return slider; });
	gw->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(this, "Horizontal Slider 4", XGLGuiSlider::Orientation::horizontal, 20, 140, 200, 16); return slider; });
	gm->AddReshapeCallback([gw](int w, int h) {
		gw->model = glm::translate(glm::mat4(), glm::vec3(w - gw->width - 20, 20, 1.0));
	});

	return;
}
