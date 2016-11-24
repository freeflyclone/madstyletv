#include "ExampleXGL.h"
#include <AntTweakBar.h>

class ATBShape : public XGLShape {
public:
	ATBShape(ExampleXGL *xgl) : pxgl(xgl), flags(0), speed(1.0), time(1.0) {
		TwInit(TW_OPENGL_CORE, NULL);
		TwBar *bar = TwNewBar("MadStyle");

		TwDefine("MadStyle color='63 63 63' label='MadStyle TV AntTweakBar Integration Testing' size='400 300'");

		// Add 'speed' to 'bar': it is a modifable (RW) variable of type TW_TYPE_DOUBLE. Its key shortcuts are [s] and [S].
		TwAddVarRW(bar, "speed", TW_TYPE_DOUBLE, &speed,
			" label='Rotation speed' min=0 max=2 step=0.01 keyIncr=s keyDecr=S help='Rotation speed (turns/second)' ");

		// Add 'wire' to 'bar': it is a modifable variable of type TW_TYPE_BOOL32 (32 bits boolean). Its key shortcut is [w].
		TwAddVarRW(bar, "wire", TW_TYPE_BOOL32, &wire,
			" label='Wireframe mode' key=w help='Toggle wireframe display mode.' ");

		// Add 'time' to 'bar': it is a read-only (RO) variable of type TW_TYPE_DOUBLE, with 1 precision digit
		TwAddVarRO(bar, "time", TW_TYPE_DOUBLE, &time, " label='Time' precision=1 help='Time (in seconds).' ");

		// Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
		TwAddVarRW(bar, "bgColor", TW_TYPE_COLOR3F, &bgColor, " label='Background color' ");

		// Add 'cubeColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR32 (32 bits color) with alpha
		TwAddVarRW(bar, "cubeColor", TW_TYPE_COLOR32, &cubeColor,
			" label='Cube color' alpha help='Color and transparency of the cube.' ");

		pxgl->projector.AddReshapeCallback(std::bind(&ATBShape::Reshape, this, _1, _2));
		pxgl->AddMouseFunc(std::bind(&ATBShape::MouseMotion, this, _1, _2, _3));

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
	}

	~ATBShape() { TwTerminate(); }
	void Draw() { TwDraw(); }
	void Reshape(int w, int h) { TwWindowSize(w, h); }

	void MouseMotion(int x, int y, int f) {
		int button = (f ^ flags);
		int action = (f & 0xF) ? 1 : 0;

		if (button) {
			button--;
			TwEventMouseButtonGLFW(button, action);
		}

		TwEventMousePosGLFW(x, y);
		flags = f;
	}

	ExampleXGL *pxgl;
	int flags;
	double speed;
	bool wire;
	double time;
	glm::vec3 bgColor, cubeColor;
};

class XGLGuiCanvasWithReshape : public XGLGuiCanvas {
public:
	XGLGuiCanvasWithReshape(int w, int h) : XGLGuiCanvas(w, h), ww(w), wh(h), wx(0), wy(0) {
		attributes.diffuseColor = { 0.0, 0.0, 0.0, 0.0 };

		SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
			wx = (int)((1.0 + x) / 2.0 * (float)ww);
			wy = (int)((1.0 - y) / 2.0 * (float)wh);

			//xprintf("%d, %d, %08X\n", wx, wy, flags);
			if (Children().size())
				xprintf("There are children\n");

			return true;
		});
	}

	// So we will know the shape of the window we're in.
	void Reshape(int w, int h) { ww = w; 	wh = h; }

	int ww, wh, wx, wy;
};


void ExampleXGL::BuildGUI() {
	XGLGuiCanvasWithReshape *gc;
	XGLGuiCanvas *child1, *child2, *child3;
	glm::mat4 translate, scale, model;

	AddGuiShape("shaders/000-simple", [&]() { return new XGLTransformer(); });

	// add the AntTweakBar shape on top of the XGLGuiCanvasWithReshape
	AddGuiShape("shaders/tex", [&]() { return new ATBShape(this); });

	AddGuiShape("shaders/000-simple", [&]() {
		gc = new XGLGuiCanvasWithReshape(projector.width, projector.height);
		gc->SetXGL(this);
		projector.AddReshapeCallback(std::bind(&XGLGuiCanvasWithReshape::Reshape, gc, _1, _2));
		return gc;
	});

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child2 = new XGLGuiCanvas(1920, 1080); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.4f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child2->model = model;
	gc->AddChild(child2);
	child2->RenderText(L"Now is the time for all good men \nto come to the aid of their country.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(640, 360); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.5 };
	gc->AddChild(child1);
	child1->RenderText(L"Really BIG text.\nReally really big text.\nSeriously big.\nSERIOUSLY! It's big.\nHuge even.");
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child3 = new XGLGuiCanvas(1280, 720); return child3; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, -0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child3->model = model;
	child3->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	child2->AddChild(child3);
	child3->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	child3->RenderText(L"Smaller canvas window inside larger one.");

	CreateShape(&guiShapes, "shaders/gui", [&]() { child1 = new XGLGuiCanvas(); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0, -0.9f, 0));
	model = glm::scale(translate, glm::vec3(0.99, 0.025, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 0.5, 0.5, 0.5, 0.3 };
	gc->AddChild(child1);
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		XGLGuiCanvas *gc = (XGLGuiCanvas *)s;
		XGLGuiCanvas *cgc = (XGLGuiCanvas *)s->Children()[0];
		static float oldX = 0.0f;

		// if mouse is down
		if (flags & 1) {
			// contain thumb inside track
			if (x < (-1 + cgc->model[0].x))
				x = -1 + cgc->model[0].x;
			if (x >(1 - cgc->model[0].x))
				x = 1 - cgc->model[0].x;

			if (y < (-1 + cgc->model[1].y))
				y = -1 + cgc->model[1].y;
			if (y >(1 - cgc->model[1].y))
				y = 1 - cgc->model[1].y;

			// adjust the X,Y coordinates of translation row of thumb's model matrix
			cgc->model[3].x = x;
			cgc->model[3].y = y;

			mouseCaptured = s;

			if (x != oldX) {
				xprintf("Slider: %0.5f\n", (1.0f + (x / (1.0f - cgc->model[0].x))) / 2.0f);
				oldX = x;
			}
		}
		else
			mouseCaptured = NULL;

		gc->childEvent = false;
		return true;
	});

	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLGuiCanvas(); return child2; });
	scale = glm::scale(glm::mat4(), glm::vec3(0.02, 1, 1.0));
	child2->model = scale;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	child1->AddChild(child2);
	child2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		// parent is known to be XGLGuiCanvas.  We just did that above
		XGLGuiCanvas *p = (XGLGuiCanvas *)s->parent;
		p->childEvent = true;
		return false;
	});
}
