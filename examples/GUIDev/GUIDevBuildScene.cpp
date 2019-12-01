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

class XGLGuiTextEdit : public XGLGuiCanvas {
public:
	XGLGuiTextEdit(XGL *xgl, std::string name, int x, int y, int w, int h) : XGLGuiCanvas(xgl, w, h){
		SetName(name);
		attributes.ambientColor = XGLColors::black;
		attributes.ambientColor.a = 0.7f;
		attributes.diffuseColor = XGLColors::black;

		AddChildShape("shaders/ortho", [xgl, this, name, x, y]() { label = new XGLGuiLabel(xgl, name, x, y); return label; });
		label->model = glm::translate(glm::mat4(), glm::vec3(-(label->labelWidth + label->labelPadding), 0, 0.0));

		model = glm::translate(glm::mat4(), glm::vec3(x + (label->labelWidth + label->labelPadding), y, 0));
		SetMouseFunc([xgl, this](float x, float y, int flags){
			static int oldFlags = 0;

			if (flags & 1)
				CaptureMouse();
			else
				ReleaseMouse();

			// if a button has changed state...
			if (flags ^ oldFlags) {
				//... if the left button is presently down...
				if (flags & 1) {
					// ...if I don't have focus...
					if (!HasFocus()) {
						//...but someone else does, take it from them...
						if (Focused()) {
							xprintf("%s released keyboard\n", Focused()->Name().c_str());
							Focused()->ReleaseKeyboard();
						}
						// ...and then claim it for myself
						CaptureKeyboard();
						xprintf("%s captured keyboard\n", Name().c_str());
					}
				}
				oldFlags = flags;
			}

			return true;
		});
	}

private:
	XGLGuiLabel *label;
};

class XGLRoundedTexQuad : public XGLTexQuad {
public:
	XGLRoundedTexQuad() : XGLTexQuad() {};
	XGLRoundedTexQuad(std::string fileName) : XGLTexQuad(fileName) {};
	XGLRoundedTexQuad(int width, int height, int channels, GLubyte *img, bool flipColors = false) : XGLTexQuad(width, height, channels, img, flipColors) {};
	XGLRoundedTexQuad(int w, int h, int c) : XGLTexQuad(w, h, c), width(w), height(h), channels(c) {
		xprintf("%s()\n", __FUNCTION__);

		bufferSize = width*height*channels;
		buffer = new GLbyte[bufferSize]();
		memset(buffer, 0, bufferSize);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texIds[0]);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, buffer);
		GL_CHECK("glGetTexSubImage2D() didn't work");
	};

	GLbyte* buffer{ nullptr };
	int bufferSize{ 0 };
	int width, height, channels;
};

void ExampleXGL::BuildScene() {
	XGLShape *shape, *rrTexQuad;
	XGLGuiManager *gm = GetGuiManager();
	XGLGuiWindow *gw;
	XGLGuiTextEdit *gte;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	// Initialize the Camera matrix
	glm::vec3 cameraPosition(-3.75, -6.25, 3.75);
	glm::vec3 cameraDirection = glm::normalize(cameraPosition*-1.0f);
	glm::vec3 cameraUp = { 0, 0, 1 };
	camera.Set(cameraPosition, cameraDirection, cameraUp);

	AddShape("shaders/mono", [&]() { rrTexQuad = new XGLRoundedTexQuad(640, 360, 1); return rrTexQuad; });
	rrTexQuad->attributes.ambientColor = XGLColors::magenta;
	rrTexQuad->attributes.diffuseColor = XGLColors::yellow;
	rrTexQuad->model = glm::scale(glm::mat4(), glm::vec3(1.77777, 1, 1));

	gm->AddChildShape("shaders/ortho-tex", [&]() { gw = new XGLGuiWindow(this, "TextWindow", 480, 20, 360, 300); return gw; });
	gw->attributes.diffuseColor = XGLColors::white;
	gw->attributes.ambientColor = { 0.8, 0.8, 0.8, 0.5 };


	gw->SetPenPosition(10, 20);
	gw->RenderText("Container for XGLGuiTextEdit fields.\n", 20);

	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 1", 20, 40, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 2", 20, 80, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 3", 20, 120, 200, 24); return gte; });
	gw->AddChildShape("shaders/ortho-tex", [&gte, this]() { gte = new XGLGuiTextEdit(this, "Text Edit 4", 20, 160, 200, 24); return gte; });
}
