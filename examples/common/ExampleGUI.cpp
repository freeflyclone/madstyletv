#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLGuiCanvasWithReshape *gc;
	XGLGuiCanvas *child1, *child2, *child3;
	glm::mat4 translate, scale, model;

	AddGuiShape("shaders/000-simple", [&]() { return new XGLTransformer(); });

	// add the AntTweakBar shape on top of the XGLGuiCanvasWithReshape
	AddGuiShape("shaders/tex", [&]() { return new XGLAntTweakBar(this); });

	AddGuiShape("shaders/000-simple", [&]() {
		gc = new XGLGuiCanvasWithReshape(this, projector.width, projector.height);
		gc->SetXGL(this);
		projector.AddReshapeCallback(std::bind(&XGLGuiCanvasWithReshape::Reshape, gc, _1, _2));
		return gc;
	});
	// make sure this shape is invisible. MUST be done after the above function, not inside it.
	gc->SetColor({ 0.0, 0.0, 0.0, 0.0 });

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child2 = new XGLGuiCanvas(this, 1920, 1080); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.4f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child2->model = model;
	gc->AddChild(child2);
	child2->RenderText(L"Now is the time for all good men \nto come to the aid of their country.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(this, 640, 360); return child1; });
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

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child3 = new XGLGuiCanvas(this, 1280, 720); return child3; });
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

	CreateShape(&guiShapes, "shaders/gui", [&]() { child1 = new XGLGuiCanvas(this); return child1; });
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

	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLGuiCanvas(this); return child2; });
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

	//AddGuiShape("shaders/ortho-tex", [&]() { child1 = new XGLGuiCanvas(this, 440, 20, 360, 640); return child1; });
	//child1->RenderText(L"Test");
}
