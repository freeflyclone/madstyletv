#include "ExampleXGL.h"

namespace {
	XGLGuiCanvas *CreateSliderTrack(XGL *xgl, XGLGuiCanvas *container, std::string name, int x, int y, int w, int h) {
		XGLGuiCanvas *gc;
		container->AddChildShape("shaders/ortho", [xgl, &gc, x, y, w, h]() { gc = new XGLGuiCanvas(xgl, w, h, false); return gc; });
		gc->SetName(name, false);
		gc->attributes.diffuseColor = { 1, 1, 1, 0.1 };
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		return gc;
	}

	XGLGuiCanvas *CreateSliderGroove(XGL *xgl, XGLGuiCanvas *track, int x, int y, int w, int h) {
		XGLGuiCanvas *gc;
		track->AddChildShape("shaders/ortho", [xgl, &gc, x, y, w, h]() { gc = new XGLGuiCanvas(xgl, w, h, false); return gc; });
		gc->attributes.diffuseColor = white;
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		return gc;
	}

	XGLGuiCanvas *CreateSliderThumb(XGL *xgl, XGLGuiCanvas *track, int x, int y, int w, int h){
		XGLGuiCanvas *gc;
		track->AddChildShape("shaders/ortho-rgb", [xgl, &gc, x, y, w, h]() { gc = new XGLGuiCanvas(xgl, w, h, false); return gc; });
		gc->AddTexture(pathToAssets + "/assets/button-large.png");
		gc->attributes.ambientColor = { 0, 0, 0, 0 };
		gc->Reshape(0, 0, w, h);
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		return gc;
	}

	XGLGuiCanvas *CreateSliderLabel(XGL *xgl, XGLGuiCanvas *track, std::string name,  int x, int y) {
		XGLGuiCanvas *gc;
		font.SetPixelSize(12);
		int fontHeight = font.MeasureFontHeight();
		int baselineHeight = font.MeasureBaselineHeight();
		int labelPadding = 8;
		int labelWidth = font.MeasureStringWidth(name) + labelPadding;
		int labelHeight = fontHeight + labelPadding;

		track->AddChildShape("shaders/ortho-tex", [xgl, &gc, x, y, labelWidth, labelHeight]() { gc = new XGLGuiCanvas(xgl, labelWidth, labelHeight); return gc; });
		gc->SetName("Label", false);
		gc->attributes.diffuseColor = white;
		gc->attributes.ambientColor = { 1, 1, 1, 0.1 };
		gc->model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		gc->SetPenPosition(labelPadding / 2, labelHeight - (baselineHeight + (labelPadding / 2)));
		gc->RenderText(name.c_str(), 12);

		return gc;
	}

	XGLGuiCanvas *CreateHorizontalSlider(XGL *xgl, XGLGuiCanvas *container, std::string name, int x, int y, int width) {
		XGLGuiCanvas *gc, *g4;
		int trackHeight = 16;
		font.SetPixelSize(12);
		int fontHeight = font.MeasureFontHeight();
		int labelPadding = 8;
		int labelWidth = font.MeasureStringWidth(name) + labelPadding;
		int labelHeight = fontHeight + labelPadding;

		gc = CreateSliderTrack(xgl, container, name, x, y, width, 16);
		g4 = CreateSliderGroove(xgl, gc, trackHeight / 4, trackHeight / 2, width - (trackHeight / 2), 1);
		g4 = CreateSliderThumb(xgl, gc, 0, 0, trackHeight, trackHeight);
		g4 = CreateSliderLabel(xgl, gc, name, -(labelWidth + labelPadding), (trackHeight / 2) - (labelHeight / 2));

		// since the label is to the left of the slider, offset the whole thing by its measured width
		gc->model *= glm::translate(glm::mat4(), glm::vec3(labelWidth + labelPadding, 0, 0));

		gc->SetMouseFunc([xgl, gc](float x, float y, int flags){
			if (flags & 1) {
				XGLGuiCanvas *slider = (XGLGuiCanvas *)(gc->Children()[1]);

				// adjust mouse Y coordinate to center it in the slider window
				float offsetX = x - (slider->width / 2.0f);
				// constrain mouse X coordinate to dimensions of track
				float xLimited = (offsetX<0) ? 0 : (offsetX>(gc->width - slider->width)) ? (gc->width - slider->width) : offsetX;
				static float previousXlimited = 0.0;

				xprintf("HorizontalSlider: %0.0f\n", xLimited);

				if (xLimited != previousXlimited) {
					slider->model = glm::translate(glm::mat4(), glm::vec3(xLimited, 0.0, 0.0));
					previousXlimited = xLimited;
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

	XGLGuiCanvas *CreateVerticalSlider(XGL *xgl, XGLGuiCanvas *container, std::string name, int x, int y, int height) {
		XGLGuiCanvas *gc, *g4;
		int trackWidth = 16;
		font.SetPixelSize(12);
		int fontHeight = font.MeasureFontHeight();
		int baselineHeight = font.MeasureBaselineHeight();
		int labelPadding = 8;
		int labelWidth = font.MeasureStringWidth(name) + labelPadding;
		int labelHeight = fontHeight + labelPadding;

		gc = CreateSliderTrack(xgl, container, name, x, y, 16, height);
		g4 = CreateSliderGroove(xgl, gc, trackWidth / 2, trackWidth / 4, 1, height - (trackWidth / 2));
		g4 = CreateSliderThumb(xgl, gc, 0, height - trackWidth, trackWidth, trackWidth);
		g4 = CreateSliderLabel(xgl, gc, name, (trackWidth /2) + x - (labelWidth + labelPadding), height + (labelHeight / 2));

		gc->SetMouseFunc([xgl, gc](float x, float y, int flags){
			if (flags & 1) {
				XGLGuiCanvas *slider = (XGLGuiCanvas *)(gc->Children()[1]);
				// adjust mouse Y coordinate to center it in the slider window
				float offsetY = y - (slider->height / 2.0f);
				// constrain mouse Y coordinate to dimensions of track
				float yLimited = (offsetY<0) ? 0 : (offsetY>(gc->height - slider->height)) ? (gc->height - slider->height) : offsetY;
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

	void CreateGuiWindows(XGL *xgl) {
		XGLGuiCanvas *g;
		XGLGuiManager *gm;

		xgl->AddGuiShape("shaders/ortho", [&gm, xgl]() { gm = new XGLGuiManager(xgl); return gm; });

		gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(xgl, 88, 500); return g; });
		g->SetName("SliderWindow", false);
		g->model = glm::translate(glm::mat4(), glm::vec3(20, 20, 0));
		g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };

		CreateVerticalSlider(xgl, g, "Roll Rate", 36, 20, 400);

		gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(xgl, 360, 180); return g; });
		g->SetName("HorizontalSlidersWindow", false);
		g->model = glm::translate(glm::mat4(), glm::vec3(0, 60, 0));
		g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
		gm->AddReshapeCallback([g](int w, int h) {
			g->model = glm::translate(glm::mat4(), glm::vec3(w - g->width - 20, 20, 1.0));
		});

		CreateHorizontalSlider(xgl, g, "Horizontal Slider 1", 20, 20, 200);
		CreateHorizontalSlider(xgl, g, "Horizontal Slider 2", 20, 60, 200);
		CreateHorizontalSlider(xgl, g, "Horizontal Slider 3", 20, 100, 200);
		CreateHorizontalSlider(xgl, g, "Horizontal Slider 4", 20, 140, 200);
	}
};

void ExampleXGL::BuildGUI() {
	CreateGuiWindows(this);

	return;
}
