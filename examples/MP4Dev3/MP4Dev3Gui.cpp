#include "ExampleXGL.h"

XGLGuiSlider *scrollSlider;

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;
	XGLGuiWindow *gw;
	XGLGuiLabel *gl;

	AddGuiShape("shaders/ortho", [&]() { gm = new XGLGuiManager(this); return gm; });

	gm->AddChildShape("shaders/ortho", [&]() { gw = new XGLGuiWindow(this, "HorizontalSliderWindow", 20, 20, 460, 180); return gw; });
	gw->AddChildShape("shaders/ortho", [&]() { return new XGLGuiSlider(this, "Frames/Second", XGLGuiSlider::Orientation::horizontal, 20, 20, 200, 16); });
	gw->AddChildShape("shaders/ortho", [&]() { gl = new XGLGuiLabel(this, "FPSValue", 340, 16); return gl; });
	gl->SetName("FPSValue", false);

	gm->AddChildShape("shaders/ortho", [&]() { 
		scrollSlider = new XGLGuiSlider(this, "File Seek", XGLGuiSlider::Orientation::horizontal, 20, 900, 1700, 16); 
		return scrollSlider; 
	});
	gm->AddReshapeCallback([gm](int w, int h) {
		scrollSlider->model = glm::translate(glm::mat4(), glm::vec3(120, h - 40, 0.0));
	});

	return;
}
