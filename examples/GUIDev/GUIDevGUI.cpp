#include "ExampleXGL.h"

class Gui : public XGLShape {
public:
	Gui(XGL *xgl, int x, int y, int w, int h) : pxgl(xgl), xOrig(x), yOrig(y), width(w), height(h) {
		SetName("Gui");
		const XGLColor white = { 1, 1, 1, 1 };

		v.push_back({ { 0, 0, 0 }, { 0, 0 }, {}, white });
		v.push_back({ { 0, h, 0 }, { 0, 1 }, {}, white });
		v.push_back({ { w, 0, 0 }, { 1, 0 }, {}, white });
		v.push_back({ { w, h, 0 }, { 1, 1 }, {}, white });

		idx.push_back(0);
		idx.push_back(1);
		idx.push_back(2);
		idx.push_back(3);

		attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.5 };

		// our base class is a dimensionless XGLTexQuad with no texture map
		// but we want a texture map that's easily accessible for GUI work
		// so create a host memory buffer and add it to our base XGLTexQuad
		int size = (width - xOrig)*(height - yOrig);
		if ((buffer = new GLubyte[size]()) == NULL)
			throwXGLException("failed to allocate a buffer for the XGLGuiCanvas");

		memset(buffer, 63, size);
		AddTexture(width-xOrig, height-yOrig, 1, buffer);

		XInput::XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
			const bool isDown = (flags & 0x8000) == 0;
			const bool isRepeat = (flags & 0x4000) != 0;

			if (isDown && pxgl->GuiIsActive())
				pxgl->RenderGui(false);
			else if (isDown)
				pxgl->RenderGui(true);
		};

		pxgl->AddKeyFunc('`', PresentGuiCanvas);
		pxgl->AddKeyFunc('~', PresentGuiCanvas);

		pxgl->projector.AddReshapeCallback(std::bind(&Gui::Reshape, this, _1, _2));
	}

	void Reshape(int w, int h) {
		model = glm::translate(glm::ortho(0.0f, (float)w, (float)h, 0.0f), glm::vec3((float)xOrig, (float)yOrig, 0.0));
	}

	void Draw() {
		xprintf("Draw()\n");
		glEnable(GL_BLEND);
		GL_CHECK("glEnable(GL_BLEND) failed");
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		GL_CHECK("glBlendFunc() failed");

		glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)(idx.size()), XGLIndexType, 0);
		GL_CHECK("glDrawElements() failed");

		glDisable(GL_BLEND);
		GL_CHECK("glDisable(GL_BLEND) failed");
	}

	XGL *pxgl;
	GLubyte *buffer;
	int xOrig, yOrig;
	int width, height;
};

void ExampleXGL::BuildGUI() {
	AddGuiShape("shaders/ortho-tex", [&]() { return new Gui(this, 20, 20, 640, 360); });
}
