#include "ExampleXGL.h"

namespace {
	XGLGuiCanvas *CreateVerticalSlider(XGL *xgl, XGLGuiCanvas *container, int x, int y, int height) {
		XGLGuiCanvas *gc, *g4;

		container->AddChildShape("shaders/ortho", [xgl, &gc, x, y, height]() { gc = new XGLGuiCanvas(xgl, 16, height); return gc; });
		gc->SetName("VerticalSlider");
		gc->attributes.diffuseColor = { 1, 1, 1, 0.1 };
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));

		gc->AddChildShape("shaders/ortho", [xgl, &g4, height]() { g4 = new XGLGuiCanvas(xgl, 1, height - 8, false); return g4; });
		g4->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
		g4->model = glm::translate(glm::mat4(), glm::vec3(8.0, 4.0, 0.0));

		gc->AddChildShape("shaders/ortho-rgb", [xgl, &g4, height]() { g4 = new XGLGuiCanvas(xgl, 16, 16, false); return g4; });
		g4->AddTexture(pathToAssets + "/assets/button.png");
		g4->attributes.diffuseColor = { 1.0, 0.0, 1.0, 0.8 };
		g4->Reshape(0, 0, 16, 16);
		g4->model = glm::translate(glm::mat4(), glm::vec3(0.0, height - 16, 0.0));

		gc->SetMouseFunc([xgl, gc](float x, float y, int flags){
			if (flags & 1) {
				XGLGuiCanvas *slider = (XGLGuiCanvas *)(gc->Children()[1]);
				// constrain mouse Y coordinate to dimensions of track
				float yLimited = (y<0) ? 0 : (y>(gc->height - slider->height)) ? (gc->height - slider->height) : y;
				static float previousYlimited = 0.0;

				if (yLimited != previousYlimited) {
					slider->model = glm::translate(glm::mat4(), glm::vec3(0.0, yLimited, 0.0));
					previousYlimited = yLimited;
				}
				xgl->mouseCaptured = gc;
				gc->SetHasMouse(true);
			}
			else {
				xgl->mouseCaptured = NULL;
				gc->SetHasMouse(false);
			}
			return true;
		});

		return gc;
	}

	void CreateGuiWindows(XGL *xgl, XGLGuiManager *gm) {
		XGLGuiCanvas *g;

		gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(xgl, 300, 500); return g; });
		g->model = glm::translate(glm::mat4(), glm::vec3(60, 60, 0));
		g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };

		CreateVerticalSlider(xgl, g, 36, 20, 400);
		CreateVerticalSlider(xgl, g, 144, 20, 400);
		CreateVerticalSlider(xgl, g, 248, 20, 400);
	}
};

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;

	AddGuiShape("shaders/ortho", [&gm,this]() { gm = new XGLGuiManager(this); return gm; });

	CreateGuiWindows(this, gm);

	return;
}
