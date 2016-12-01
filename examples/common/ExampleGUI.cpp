#include "ExampleXGL.h"

class GuiManager : public XGLTransformer {
public:
	typedef std::function<void(XGLGuiCanvas *, int, int)> ReshapeFunc;
	typedef std::pair<XGLGuiCanvas *, ReshapeFunc> ReshapePair;
	typedef std::vector<ReshapePair> ReshapeCallbackList;

	GuiManager(XGL *xgl, bool addTexture = false) : pxgl(xgl), padding(20) {
		SetName("GuiManager");

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

		xgl->projector.AddReshapeCallback(std::bind(&GuiManager::Reshape, this, _1, _2));
	}

	void AddReshapeCallback(XGLGuiCanvas *s, ReshapeFunc fn) {
		reshapeCallbacks.push_back(ReshapePair(s, fn));
	}

	void Reshape(int w, int h) {
		for (rc = reshapeCallbacks.begin(); rc < reshapeCallbacks.end(); rc++)
			(rc->second)(rc->first, w, h);
	}

	XGL *pxgl;
	int padding;
	ReshapeCallbackList reshapeCallbacks;
	ReshapeCallbackList::iterator rc;
};

void ExampleXGL::BuildGUI() {
	GuiManager *gm;
	XGLGuiCanvas *g, *g2;

	AddGuiShape("shaders/ortho", [&]() { gm = new GuiManager(this); return gm; });

	AddGuiShape("shaders/ortho", [&]() { g = new XGLGuiCanvas(this, 1, 1, false); return g; });
	g->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
	g->SetMouseFunc([&](XGLShape *s, float x, float y, int flags){
		xprintf("In %s(%0.0f,%0.0f)\n", s->name.c_str(), x, y);

		if (flags & 1)
			mouseCaptured = (XGLGuiCanvas *)s;
		else
			mouseCaptured = NULL;

		return true;
	});
	gm->AddReshapeCallback(g, [](XGLGuiCanvas *gc, int w, int h) {
		xprintf("%d,%d in %s\n", w, h, gc->name.c_str());
		gc->width = w;
		gc->height = h;
		gc->Reshape(0, 0, w, h);
	});

	CreateShape(&guiShapes, "shaders/ortho", [&]() { g2 = new XGLGuiCanvas(this, 360, 640); return g2; });
	g2->model = glm::translate(glm::mat4(), glm::vec3(800, 20, 0));
	g2->attributes.diffuseColor = { 1.0, 0.0, 1.0, 0.1 };
	g2->SetMouseFunc([&](XGLShape *s, float x, float y, int flags){
		xprintf("In %s(%0.0f,%0.0f)\n", s->name.c_str(), x, y);
		if (flags & 1)
			mouseCaptured = (XGLGuiCanvas *)s;
		else
			mouseCaptured = NULL;
		return true;
	});
	gm->AddReshapeCallback(g2, [](XGLGuiCanvas *gc, int w, int h) {
		xprintf("%d,%d in %s\n", w, h, gc->name.c_str());
		gc->model = glm::translate(glm::mat4(), glm::vec3(w - gc->width - 20, 20, 1.0));
	});
	g->AddChild(g2);

	AddGuiShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });

	return;
}
