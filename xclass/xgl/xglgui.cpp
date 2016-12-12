#include "xgl.h"

XGLGuiCanvas::XGLGuiCanvas(XGL *xgl) :
XGLTexQuad(),
buffer(NULL),
pxgl(xgl),
childEvent(false)
{
	SetName("XGLGuiCanvas");
	attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.5 };
}

XGLGuiCanvas::XGLGuiCanvas(XGL *xgl, int w, int h, bool addTexture) :
XGLTexQuad(w, h),
buffer(NULL),
pxgl(xgl),
childEvent(false)
{
	SetName("XGLGuiCanvas");
	attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.5 };

	width = w;
	height = h;
	penX = 10;
	penY = 64;

	if (addTexture) {
		// our base class is XGLTexQuad with no texture map
		// but we want a texture map that's easily accessible for GUI work
		// so create a host memory buffer and add it to our base XGLTexQuad
		if ((buffer = new GLubyte[width*height]()) == NULL)
			throwXGLException("failed to allocate a buffer for the XGLGuiCanvas");

		memset(buffer, 0, width*height);
		AddTexture(width, height, 1, buffer);
	}
}

XGLGuiCanvas::~XGLGuiCanvas() {
	if (buffer != NULL)
		delete buffer;
}

void XGLGuiCanvas::AddChildShape(std::string shaderName, XGLNewShapeLambda fn){
	AddChild(pxgl->CreateShape(&(pxgl->guiShapes), shaderName, fn));
}

void XGLGuiCanvas::SetMouseFunc(XGLGuiCanvas::MouseFunc fn){
	mouseFunc = fn;
}

bool XGLGuiCanvas::MouseEvent(float x, float y, int flags) {
	bool retVal = false;

	if (mouseFunc) {
		retVal = mouseFunc(x, y, flags);

		for (auto fn : mouseEventListeners)
			fn(x, y, flags);
	}
	return retVal;
}

void XGLGuiCanvas::AddMouseEventListener(XGLGuiCanvas::MouseEventListener fn) {
	mouseEventListeners.push_back(fn);
}

void XGLGuiCanvas::MeasureFontMetrics(std::string name) {
	font.SetPixelSize(pixelSize);
	fontHeight = font.MeasureFontHeight();
	baselineHeight = font.MeasureBaselineHeight();
	labelPadding = pixelSize * 2 / 3;
	labelWidth = font.MeasureStringWidth(name) + labelPadding;
	labelHeight = fontHeight + labelPadding;
}

void XGLGuiCanvas::RenderText(std::string text, int pixelSize) {
	std::wstringstream ws;
	ws << text.c_str();
	RenderText(ws.str(), pixelSize);
}

void XGLGuiCanvas::RenderText(std::wstring text, int pixelSize) {
	font.SetPixelSize(pixelSize);
	font.RenderText(text, buffer, width, height, &penX, &penY);

	// this should probably be done with just the rectangle of the line in question
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texIds[0]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, buffer);
	GL_CHECK("glGetTexImage() didn't work");
}

void XGLGuiCanvas::Fill(GLubyte val)  {
	memset(buffer, val, width*height);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texIds[0]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, buffer);
	GL_CHECK("glGetTexImage() didn't work");
}

XGLGuiManager::XGLGuiManager(XGL *xgl, bool addTexture) : XGLGuiCanvas(xgl), pxgl(xgl), padding(20) {
	SetName("XGLGuiManager",false);

	XInput::XInputKeyFunc PresentGuiCanvas = [this](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && pxgl->GuiIsActive())
			pxgl->RenderGui(false);
		else if (isDown)
			pxgl->RenderGui(true);
	};

	pxgl->AddKeyFunc('`', PresentGuiCanvas);
	pxgl->AddKeyFunc('~', PresentGuiCanvas);

	xgl->projector.AddReshapeCallback([this](int w, int h) {
		for (auto rc : reshapeCallbacks)
			rc(w, h);
	});
}

void XGLGuiManager::AddReshapeCallback(ReshapeCallback fn) {
	reshapeCallbacks.push_back(fn);
}

XGLGuiWindow::XGLGuiWindow(XGL *xgl, std::string name, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h) {
	SetName(name, false);
	model = glm::translate(glm::mat4(), glm::vec3(x, y, 0));
	attributes.ambientColor = { 1.0, 1.0, 1.0, 0.1 };
}

XGLGuiSlider::XGLGuiSlider(XGL *xgl, std::string name, Orientation o, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h), orientation(o), position(0.0f) {
	SetName(name, false);
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
			float limit = (float)((orientation == vertical) ? (height - thumb->height) : (width - thumb->width));
			float posLimited = (pos<0) ? 0 : (pos>(limit)) ? (limit) : pos;
			static float previousPos = 0.0;

			if (posLimited != previousPos) {
				if (orientation == vertical)
					thumb->model = glm::translate(glm::mat4(), glm::vec3(0.0, posLimited, 0.0));
				else
					thumb->model = glm::translate(glm::mat4(), glm::vec3(posLimited, 0.0, 0.0));
				previousPos = posLimited;
				position = posLimited / limit;
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

void XGLGuiSlider::AdjustForOrientation(Orientation orientation, int x, int y, int w, int h) {
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
		labelY = -(labelHeight / 2) + (h / 2);
		labelOffset = glm::translate(glm::mat4(), glm::vec3(labelWidth + h, 0.0, 0.0));
	}
}
