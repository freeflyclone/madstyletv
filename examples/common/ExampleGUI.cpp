#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLShape *shape;
	XGLGuiCanvas *child1,*child2,*child3;
	//XGLShape *child3;
	glm::mat4 translate, scale, model;

	// add a GUI layer. This is essentially the same as the world layer.
	// With just a "model" transform in the vertex shader, it becomes 2D
	AddGuiShape("shaders/gui", [&]() { shape = new XGLShape(); return shape; });

	XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && GuiIsActive())
			RenderGui(false);
		else if (isDown)
			RenderGui(true);
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);

	CreateShape(&guiShapes, "shaders/gui", [&]() { child1 = new XGLGuiCanvas(); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 0.5, 0.5, 0.5, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child2 = new XGLGuiCanvas(1920,1080); return child2; });
	child1->AddChild(child2);
	child2->RenderText(L"Now is the time for all good men \nto come to the aid of their country.");


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

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(640, 360); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 0.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->RenderText(L"Really BIG text.\nReally really big text.\nSeriously big.\nSERIOUSLY! It's big.\nHuge even.");
	child1->SetMouseFunc([&](XGLShape *s, float x, float y, int flags) {
		xprintf("In MouseFunc() for %s : %0.4f, %0.4f\n", s->name.c_str(), x, y);
		return true;
	});

	CreateShape(&guiShapes, "shaders/gui", [&]() { child1 = new XGLGuiCanvas(); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0, -0.9f, 0));
	model = glm::scale(translate, glm::vec3(0.99, 0.025, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 0.5, 0.5, 0.5, 0.3 };
	GetGuiRoot()->AddChild(child1);
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
