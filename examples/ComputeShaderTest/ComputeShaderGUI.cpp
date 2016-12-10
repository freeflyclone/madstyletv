#include "ExampleXGL.h"
#include <filesystem>

/*
** XGLGuiSlider: define the essence of a GUI slider control, that can be either vertical or horizontal.
**
** Much of the layout is defaulted to arbitrary choices according to my personal preferences.  My
** preferences have been influenced by others work. I didn't invent anything here. I'm just emulating
** what I've seen in ALL the major desktop OS GUI frameworks.
**
** The goal here is the minimum amount of code necessary to provide visibly acceptable layout with 
** unsurprising event response.
*/
class XGLGuiSlider : public XGLGuiCanvas {
public:
	enum Orientation {
		vertical,
		horizontal
	};

	XGLGuiSlider(XGL *xgl, std::string name, Orientation o, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h), orientation(o) {
		model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		attributes.ambientColor = { 1, 1, 1, 0.1 };

		// twiddle the layout variables according to an arbitrarilly chosen font size
		MeasureFontMetrics(name);
		AdjustForOrientation(orientation, x, y, w, h);

		// the "groove" is just a line down the middle
		AddChildShape("shaders/ortho", [xgl, this, x, y, w, h]() { groove = new XGLGuiCanvas(xgl, grooveWidth, grooveHeight, false); return groove; });
		groove->attributes.ambientColor = white;
		groove->model = glm::translate(glm::mat4(), glm::vec3(grooveOffset, grooveOffset, 0.0));

		// the thumb is the thingy that moves according to mouse position
		AddChildShape("shaders/ortho-rgb", [xgl, this, x, y, w, h]() { thumb = new XGLGuiCanvas(xgl, w, h, false); return thumb; });
		thumb->AddTexture(pathToAssets + "/assets/button-large.png");
		thumb->attributes.ambientColor = { 0, 0, 0, 0 };
		thumb->Reshape(0, 0, thumbSize, thumbSize);
		thumb->width = thumbSize;
		thumb->height = thumbSize;
		thumb->model = glm::translate(glm::mat4(), glm::vec3(thumbX, thumbY, 0.0));

		// the label is, well, the label.
		AddChildShape("shaders/ortho-tex", [xgl, this, x, y]() { label = new XGLGuiCanvas(xgl, labelWidth, labelHeight); return label; });
		label->SetName("Label", false);
		label->attributes.diffuseColor = white;
		label->attributes.ambientColor = { 1, 1, 1, 0.1 };
		label->SetPenPosition(labelPadding / 2, labelHeight - (baselineHeight + (labelPadding / 2)));
		label->RenderText(name.c_str(), pixelSize);
		label->model = glm::translate(glm::mat4(), glm::vec3(labelX, labelY, 0.0));

		// we move our base coordinates to the right by the width of the label if we're horizontal (left side label)
		model *= labelOffset;

		// the "position" of the slider is independent of orientation, so behave accordingly when moving the thumb
		SetMouseFunc([xgl, this](float x, float y, int flags){
			if (flags & 1) {
				float pos = orientation == vertical ? y : x;
				float limit = orientation == vertical ? (height - thumb->height) : (width - thumb->width);
				float posLimited = (pos<0) ? 0 : (pos>(limit)) ? (limit) : pos;
				static float previousPos = 0.0;

				if (posLimited != previousPos) {
					if (orientation == vertical)
						thumb->model = glm::translate(glm::mat4(), glm::vec3(0.0, posLimited, 0.0));
					else
						thumb->model = glm::translate(glm::mat4(), glm::vec3(posLimited, 0.0, 0.0));
					previousPos = posLimited;
				}
				xgl->mouseCaptured = this;
				SetHasMouse(true);
			}
			else {
				xgl->mouseCaptured = NULL;
				SetHasMouse(false);
			}
			return true;
		});
	}

	void MeasureFontMetrics(std::string name) {
		font.SetPixelSize(pixelSize);
		fontHeight = font.MeasureFontHeight();
		baselineHeight = font.MeasureBaselineHeight();
		labelPadding = pixelSize * 2 / 3;
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
	Orientation orientation;
	static const int pixelSize = 12;
};

class XGLGuiWindow : public XGLGuiCanvas {
public:
	XGLGuiWindow(XGL *xgl, std::string name, int x, int y, int w, int h) : XGLGuiCanvas (xgl, w, h) {
		SetName(name, false);
		model = glm::translate(glm::mat4(), glm::vec3(x, y, 0));
		attributes.ambientColor = { 1.0, 1.0, 1.0, 0.1 };
	}
};

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
