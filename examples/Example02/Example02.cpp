/**************************************************************
** Example03: demonstrate instantiation of a "ground"
** plane and multiple toruses with a lighting shader,
** transformations, animation functions, and child object
** chains
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape *shape, *child1, *child2, *child3, *child4;
	glm::mat4 rotate, translate;

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(5.0f, 1.0f, 32, 24); return shape; });
	shape->attributes.diffuseColor = { 0.005, 0.005, 0.005, 1 };
	shape->SetTheFunk([&](XGLShape *s, float clock) {
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 60.0f, glm::vec3(1.0f, 0.0f, 0.0f));
		s->model = rotate;
	});
	CreateShape("shaders/simple", [&](){ child3 = new XGLTransformer(); return child3; });
	child3->SetTheFunk([&](XGLShape *s, float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 45.0f, glm::vec3(0.0f, 0.0f, 1.0f));
		s->model = rotate;
	});
	shape->AddChild(child3);

	CreateShape("shaders/specular", [&](){ child4 = new XGLTorus(2.0f, 0.5f, 32, 24); return child4; });
	child4->attributes.diffuseColor = { 1.0, 0.00001, 0.00001, 1 };
	translate = glm::translate(glm::mat4(), glm::vec3(5.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child4->model = translate * rotate;

	CreateShape("shaders/simple", [&](){ child1 = new XGLTransformer(); return child1; });
	child1->SetTheFunk([&](XGLShape *s, float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / 30.0f, glm::vec3(0.0f, 0.0f, 1.0f));
		s->model = rotate;
	});

	CreateShape("shaders/specular", [&](){ child2 = new XGLTorus(0.75f, 0.25f, 32, 24); return child2; });
	child2->attributes.diffuseColor = (yellow);
	translate = glm::translate(glm::mat4(), glm::vec3(2.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child2->model = translate * rotate;

	child1->AddChild(child2);
	child4->AddChild(child1);
	child3->AddChild(child4);

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 32, 24); return shape; });
	shape->attributes.diffuseColor = (blue);
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 32, 24); return shape; });
	shape->attributes.diffuseColor = (red);
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0)) 
				 * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 32, 24); return shape; });
	shape->attributes.diffuseColor = (green);
	shape->model = glm::translate(glm::mat4(), glm::vec3(30, 0, 0)) 
				 * glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0,1,0)) 
				 * glm::scale(glm::mat4(), glm::vec3(2, 2, 2));


	XInputKeyFunc renderMod = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool wireFrameMode = false;

		if (isDown && !isRepeat){
			wireFrameMode = wireFrameMode ? false : true;
			glPolygonMode(GL_FRONT_AND_BACK, wireFrameMode ? GL_LINE : GL_FILL);
			GL_CHECK("glPolygonMode() failed.");
		}
	};

	AddKeyFunc('M', renderMod);
	AddKeyFunc('m', renderMod);
}