#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLShape *shape;
	XGLGuiCanvas *child1;
	XGLShape *child2;
	glm::mat4 translate, model;

	// add a GUI layer. This is essentially the same as the world layer.
	// With just a "model" transform in the vertex shader, it becomes 2D
	AddGuiShape("shaders/zz-gui", [&]() { shape = new XGLTransformer(); return shape; });

	XInputKeyFunc PresentGuiCanvas = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && IsGuiActive())
			RenderGui(false);
		else if (isDown)
			RenderGui(true);
	};

	AddKeyFunc('`', PresentGuiCanvas);
	AddKeyFunc('~', PresentGuiCanvas);

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(1280, 720); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->RenderText(L"Now is the time for all good men \nto come to the aid of their country.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(640, 360); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);
	child1->RenderText(L"Really BIG text.\nReally really big text.\nSeriously big.\nSERIOUSLY! It's big.\nYuge even.");

	CreateShape(&guiShapes, "shaders/gui-tex", [&]() { child1 = new XGLGuiCanvas(1920, 1080); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.48, 0.48, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.7 };
	GetGuiRoot()->AddChild(child1);

	child1->RenderText(L"This text *might* show up.\n");
	child1->RenderText(L"It might not.\n\n");
	child1->RenderText(L"Sed ut perspiciatis, unde omnis iste natus error sit voluptatem\n");
	child1->RenderText(L"accusantium doloremque laudantium, totam rem aperiam eaque\n");
	child1->RenderText(L"ipsa, quae ab illo inventore veritatis et quasi architecto\n");
	child1->RenderText(L"beatae vitae dicta sunt, explicabo.\n\n");
	child1->RenderText(L"But I must explain to you how all this mistaken idea of\n");
	child1->RenderText(L"denouncing of a pleasure and praising pain was born and\n");
	child1->RenderText(L"I will give you a complete account of the system, and\n");
	child1->RenderText(L"expound the actual teachings of the great explorer of\n");
	child1->RenderText(L"the truth, the master - builder of human happiness.\n");
	
	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
	child2->AddTexture(pathToAssets + "/assets/yellow.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
	child2->AddTexture(pathToAssets + "/assets/green.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, -0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
	child2->AddTexture(pathToAssets + "/assets/red.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.1 };
	child2->AddTexture(pathToAssets + "/assets/blue.png");
	child1->AddChild(child2);
}
