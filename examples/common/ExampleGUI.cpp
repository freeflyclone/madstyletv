#include "ExampleXGL.h"

void ExampleXGL::BuildGUI() {
	XGLShape *shape, *child1,*child2;
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

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child1 = new XGLGuiCanvas(16, 9); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	GetGuiRoot()->AddChild(child1);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTextLine(L"Now is the time for "); return child2; });
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
	child2->AddTexture(pathToAssets + "/assets/yellow.png");
	translate = glm::translate(glm::mat4(), glm::vec3(-1, 0, 0));
	child2->model = glm::scale(translate, glm::vec3(2, 0.25, 1));
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child1 = new XGLGuiCanvas(16, 9); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.4, 0.4, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.5 };
	child1->AddTexture(pathToAssets + "/assets/AndroidDemo.png");
	GetGuiRoot()->AddChild(child1);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child1 = new XGLGuiCanvas(16,9); return child1; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.5f, 0));
	model = glm::scale(translate, glm::vec3(0.48, 0.48, 1.0));
	child1->model = model;
	child1->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.5 };
	child1->AddTexture(pathToAssets + "/assets/AndroidDemo.png");
	GetGuiRoot()->AddChild(child1);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, 0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.3 };
	child2->AddTexture(pathToAssets + "/assets/yellow.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, 0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.3 };
	child2->AddTexture(pathToAssets + "/assets/green.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(-0.5, -0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.3 };
	child2->AddTexture(pathToAssets + "/assets/red.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTexQuad(); return child2; });
	translate = glm::translate(glm::mat4(), glm::vec3(0.5, -0.5, 0));
	model = glm::scale(translate, glm::vec3(0.5, 0.5, 1.0));
	child2->model = model;
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 0.3 };
	child2->AddTexture(pathToAssets + "/assets/blue.png");
	child1->AddChild(child2);

	CreateShape(&guiShapes, "shaders/zz-gui", [&]() { child2 = new XGLTextLine(L" e f g h i "); return child2; });
	child2->attributes.diffuseColor = { 1.0, 1.0, 1.0, 1.0 };
	child2->AddTexture(pathToAssets + "/assets/yellow.png");
	translate = glm::translate(glm::mat4(), glm::vec3(-1, 0, 0));
	child2->model = glm::scale(translate, glm::vec3(2, 0.25, 1));
	GetGuiRoot()->AddChild(child2);


}
