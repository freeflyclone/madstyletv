/**************************************************************
** GUIDevBuildScene.cpp
**
** Demonstrates interaction with the GUI stack. BuildScene()
** is called after BuildGUI() by the ExampleXGL constructor,
** so it is safe to assume it exists at this point.
**
** Here is where we demonstrate adding to the GUI stack, and
** connecting GUI stack objects to world objects for a
** given application.
**************************************************************/
#include "ExampleXGL.h"

class XGLGuiLabel : public XGLGuiCanvas {
public:
	XGLGuiLabel(XGL *xgl, std::string name, int x, int y) : XGLGuiCanvas(xgl) {
		SetName(name);
		model = glm::translate(glm::mat4(), glm::vec3(x, y, 0.0));
		attributes.ambientColor = { 1, 1, 1, 0.1 };

		// twiddle the layout variables according to an arbitrarilly chosen font size
		MeasureFontMetrics(name);

		// the actual label is a child object, because we need to measure the size of the
		// required text bounding box first, and setting the geometry shape inside the
		// constructor of a shape is problematic. (The "shader" member is not yet set,
		// therefore the "programId" isn't known yet, so OpenGL calls break)
		// This is a workaround, I'm willing to incur the technical debt for now.
		AddChildShape("shaders/ortho-tex", [xgl, this]() { label = new XGLGuiCanvas(xgl, labelWidth, labelHeight); return label; });
		label->SetName("Label", false);
		label->attributes.diffuseColor = white;
		label->attributes.ambientColor = { 1, 1, 1, 0.1 };
		label->SetPenPosition(labelPadding / 2, labelHeight - (baselineHeight + (labelPadding / 2)));
		label->RenderText(name.c_str(), pixelSize);
	};

	void MeasureFontMetrics(std::string name){
		font.SetPixelSize(pixelSize);
		fontHeight = font.MeasureFontHeight();
		baselineHeight = font.MeasureBaselineHeight();
		labelPadding = pixelSize * 2 / 3;
		labelWidth = font.MeasureStringWidth(name) + labelPadding;
		labelHeight = fontHeight + labelPadding;
	};

private:
	XGLGuiCanvas *label;
	int fontHeight, baselineHeight, labelPadding, labelWidth, labelHeight;
	static const int pixelSize = 12;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLGuiManager *gm = GetGuiManager();
	XGLGuiSlider *hs;
	XGLGuiLabel *gl;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	gm->AddChildShape("shaders/ortho-tex", [&gl, this]() { gl = new XGLGuiLabel(this, "Test XGLGuiLabel", 600, 20); return gl; });

	XGLGuiCanvas *sliders = (XGLGuiCanvas *)(gm->FindObject("HorizontalSliderWindow"));
	if (sliders != nullptr) {
		if ((hs = (XGLGuiSlider *)sliders->FindObject("Horizontal Slider 1")) != nullptr) {
			hs->AddMouseEventListener([hs](float x, float y, int flags) {
				if (hs->HasMouse()) {
					xprintf("%0.4f\n", hs->Position());
				}
			});
		}
	}
}
