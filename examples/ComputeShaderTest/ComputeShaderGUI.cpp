#include "ExampleXGL.h"

namespace {
	XGLGuiCanvas *CreateVerticalSlider(XGL *xgl, XGLGuiCanvas *container, std::string name, int x, int y, int height) {
		XGLGuiCanvas *gc, *g4;
		int trackWidth = 16;
		font.SetPixelSize(12);
		int fontHeight = font.MeasureFontHeight();
		int baselineHeight = font.MeasureBaselineHeight();
		int labelPadding = 8;
		int labelWidth = font.MeasureStringWidth(name) + labelPadding;
		int labelHeight = fontHeight + labelPadding;

		container->AddChildShape("shaders/ortho", [xgl, &gc, x, y, height, trackWidth]() { gc = new XGLGuiCanvas(xgl, trackWidth, height, false); return gc; });
		gc->SetName(name,false);
		gc->attributes.diffuseColor = { 1, 1, 1, 0.1 };
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));

		gc->AddChildShape("shaders/ortho", [xgl, &g4, height, trackWidth]() { g4 = new XGLGuiCanvas(xgl, 1, height - (trackWidth / 2), false); return g4; });
		g4->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
		g4->model = glm::translate(glm::mat4(), glm::vec3(trackWidth/2, trackWidth/4, 0.0));

		gc->AddChildShape("shaders/ortho-rgb", [xgl, &g4, height, trackWidth]() { g4 = new XGLGuiCanvas(xgl, trackWidth, trackWidth, false); return g4; });
		g4->AddTexture(pathToAssets + "/assets/button-large.png");
		g4->attributes.diffuseColor = { 1.0, 0.0, 1.0, 0.8 };
		g4->attributes.ambientColor = { 0, 0, 0, 0 };
		g4->Reshape(0, 0, trackWidth, trackWidth);
		g4->model = glm::translate(glm::mat4(), glm::vec3(0.0, height - trackWidth, 0.0));

		gc->AddChildShape("shaders/ortho-tex", [xgl, &g4, height, trackWidth, labelWidth, labelHeight]() { g4 = new XGLGuiCanvas(xgl, labelWidth, labelHeight); return g4; });
		g4->SetName("Label",false);
		g4->attributes.diffuseColor = white;
		g4->attributes.ambientColor = {1,1,1,0.1};
		g4->model = glm::translate(glm::mat4(), glm::vec3(-(labelWidth / 2) + (trackWidth / 2), height + labelHeight, 0.0));
		g4->SetPenPosition(labelPadding/2, labelHeight - (baselineHeight + (labelPadding / 2)));
		g4->RenderText(name.c_str(), 12);

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
		g->SetName("SliderWindow", false);
		g->model = glm::translate(glm::mat4(), glm::vec3(60, 60, 0));
		g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };

		CreateVerticalSlider(xgl, g, "Roll Rate", 36, 20, 400);
		CreateVerticalSlider(xgl, g, "WhosyWhat", 144, 20, 400);
		CreateVerticalSlider(xgl, g, "FooBar", 248, 20, 400);
	}
};

void ExampleXGL::BuildGUI() {
	XGLGuiManager *gm;

	AddGuiShape("shaders/ortho", [&gm,this]() { gm = new XGLGuiManager(this); return gm; });

	CreateGuiWindows(this, gm);

	return;
}
