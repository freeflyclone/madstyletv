#include "ExampleXGL.h"
#include <filesystem>

class XGLGuiSlider : public XGLGuiCanvas {
public:
	enum Orientation {
		vertical,
		horizontal
	};

	XGLGuiSlider(XGL *xgl, std::string name, Orientation orientation, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h) {
		model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		attributes.ambientColor = { 1, 1, 1, 0.1 };

		MeasureFontMetrics(name, 12);
		AdjustForOrientation(orientation, x, y, w, h);

		AddChildShape("shaders/ortho", [xgl, this, x, y, w, h]() { groove = new XGLGuiCanvas(xgl, grooveWidth, grooveHeight, false); return groove; });
		groove->attributes.ambientColor = white;
		groove->model = glm::translate(glm::mat4(), glm::vec3(grooveOffset, grooveOffset, 0.0));

		AddChildShape("shaders/ortho-rgb", [xgl, this, x, y, w, h]() { thumb = new XGLGuiCanvas(xgl, w, h, false); return thumb; });
		thumb->AddTexture(pathToAssets + "/assets/button-large.png");
		thumb->attributes.ambientColor = { 0, 0, 0, 0 };
		thumb->Reshape(0, 0, thumbSize, thumbSize);
		thumb->model = glm::translate(glm::mat4(), glm::vec3(thumbX, thumbY, 0.0));

		AddChildShape("shaders/ortho-tex", [xgl, this, x, y]() { label = new XGLGuiCanvas(xgl, labelWidth, labelHeight); return label; });
		label->SetName("Label", false);
		label->attributes.diffuseColor = white;
		label->attributes.ambientColor = { 1, 1, 1, 0.1 };
		label->SetPenPosition(labelPadding / 2, labelHeight - (baselineHeight + (labelPadding / 2)));
		label->RenderText(name.c_str(), 12);
		label->model = glm::translate(glm::mat4(), glm::vec3(labelX, labelY, 0.0));

		model *= labelOffset;
	}

	void MeasureFontMetrics(std::string name, int pixelSize) {
		font.SetPixelSize(pixelSize);
		fontHeight = font.MeasureFontHeight();
		baselineHeight = font.MeasureBaselineHeight();
		labelPadding = 8;
		labelWidth = font.MeasureStringWidth(name) + labelPadding;
		labelHeight = fontHeight + labelPadding;
	}

	void AdjustForOrientation(Orientation orientation, int x, int y, int w, int h) {
		if (orientation == vertical) {
			grooveWidth = 1;
			grooveHeight = h - w;
			grooveOffset = w / 2;
			thumbSize = w;
			thumbX = 0;
			thumbY = h - w;
			labelX = -(labelWidth / 2) + (w / 2);
			labelY = h + labelHeight;
		}
		else {
			grooveWidth = w - h;
			grooveHeight = 1;
			grooveOffset = h / 2;
			thumbSize = h;
			thumbX = 0;
			thumbY = 0;
			labelX = -labelWidth - h;
			labelY = -(labelHeight / 2) + (h/2);
			labelOffset = glm::translate(glm::mat4(), glm::vec3(labelWidth + h, 0.0, 0.0));
		}
	}
private:
	XGLGuiCanvas *groove, *thumb, *label;
	int fontHeight, baselineHeight, labelPadding, labelWidth, labelHeight;
	int grooveWidth, grooveHeight,grooveOffset,thumbSize,thumbX,thumbY,labelX, labelY;
	glm::mat4 labelOffset;
};

namespace {
		void CreateGuiWindows(XGL *xgl) {
		XGLGuiCanvas *g;
		XGLGuiManager *gm;
		XGLGuiSlider *slider;

		xgl->AddGuiShape("shaders/ortho", [&gm, xgl]() { gm = new XGLGuiManager(xgl); return gm; });

		gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(xgl, 88, 500); return g; });
		g->SetName("SliderWindow", false);
		g->model = glm::translate(glm::mat4(), glm::vec3(20, 20, 0));
		g->attributes.ambientColor = { 1.0, 1.0, 1.0, 0.1 };

		g->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(xgl, "Roll Rate", XGLGuiSlider::Orientation::vertical, 36, 20, 16, 400); return slider; });

		gm->AddChildShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(xgl, 360, 180); return g; });
		g->SetName("HorizontalSlidersWindow", false);
		g->model = glm::translate(glm::mat4(), glm::vec3(0, 60, 0));
		g->attributes.ambientColor = { 1.0, 1.0, 1.0, 0.1 };
		gm->AddReshapeCallback([g](int w, int h) {
			g->model = glm::translate(glm::mat4(), glm::vec3(w - g->width - 20, 20, 1.0));
		});

		g->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(xgl, "Horizontal Slider 1", XGLGuiSlider::Orientation::horizontal, 20, 20, 200, 16); return slider; });
		g->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(xgl, "Horizontal Slider 2", XGLGuiSlider::Orientation::horizontal, 20, 60, 200, 16); return slider; });
		g->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(xgl, "Horizontal Slider 3", XGLGuiSlider::Orientation::horizontal, 20, 100, 200, 16); return slider; });
		g->AddChildShape("shaders/ortho", [&]() { slider = new XGLGuiSlider(xgl, "Horizontal Slider 4", XGLGuiSlider::Orientation::horizontal, 20, 140, 200, 16); return slider; });
	}
};

void ExampleXGL::BuildGUI() {
	CreateGuiWindows(this);

	return;
}
