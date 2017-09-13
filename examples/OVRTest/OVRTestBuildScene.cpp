/**************************************************************
** OVRTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

// this needs to be file scope at least.  Local (to ::BuildScene) doesn't work
XGLSphere *sphere;
XGLShape  *hmdSled;
glm::vec3 wcPos = { 0.0f, 0.0f, 0.0f };

const float constSpeed1 = 60.0f * 4.0f;
const float constSpeed2 = 45.0f * 4.0f;
const float constSpeed3 = 30.0f * 4.0f;

float speed1 = constSpeed1;
float speed2 = constSpeed2;
float speed3 = constSpeed3;

void ExampleXGL::BuildScene() {
	XGLShape *shape, *child1, *child2, *child3, *child4;
	XGLShape *hmdChild;
	glm::mat4 rotate, translate;
	AddShape("shaders/specular", [&](){ shape = new XGLTorus(5.0f, 1.0f, 64, 32); return shape; });
	shape->attributes.diffuseColor = { 0.005, 0.005, 0.005, 1 };
	shape->SetAnimationFunction([shape](float clock) {
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed1, glm::vec3(1.0f, 0.0f, 0.0f));
		shape->model = rotate;
	});
	CreateShape("shaders/specular", [&](){ child3 = new XGLTransformer(); return child3; });
	child3->SetAnimationFunction([child3](float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed2, glm::vec3(0.0f, 0.0f, 1.0f));
		child3->model = rotate;
	});
	shape->AddChild(child3);

	CreateShape("shaders/specular", [&](){ child4 = new XGLTorus(2.0f, 0.5f, 64, 32); return child4; });
	child4->attributes.diffuseColor = { 1.0, 0.00001, 0.00001, 1 };
	translate = glm::translate(glm::mat4(), glm::vec3(5.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child4->model = translate * rotate;

	CreateShape("shaders/000-simple", [&](){ child1 = new XGLTransformer(); return child1; });
	child1->SetAnimationFunction([child1](float clock) {
		float translateFunction = sin(clock / 180.0f);
		glm::mat4 rotate = glm::rotate(glm::mat4(), clock / speed3, glm::vec3(0.0f, 0.0f, 1.0f));
		child1->model = rotate;
	});

	CreateShape("shaders/specular", [&](){ child2 = new XGLTorus(0.75f, 0.25f, 64, 32); return child2; });
	child2->attributes.diffuseColor = (XGLColors::yellow);
	translate = glm::translate(glm::mat4(), glm::vec3(2.0, 0, 0));
	rotate = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	child2->model = translate * rotate;

	child1->AddChild(child2);
	child4->AddChild(child1);
	child3->AddChild(child4);

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::blue);
	shape->model = glm::translate(glm::mat4(), glm::vec3(20, 0, 0));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::red);
	shape->model = glm::translate(glm::mat4(), glm::vec3(-20, 0, 0))
		* glm::scale(glm::mat4(), glm::vec3(2, 2, 2));

	AddShape("shaders/specular", [&](){ shape = new XGLTorus(3.0f, 0.5f, 64, 32); return shape; });
	shape->attributes.diffuseColor = (XGLColors::green);
	shape->model = glm::translate(glm::mat4(), glm::vec3(30, 0, 0))
		* glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0, 1, 0))
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

	AddShape("shaders/000-simple", [&]() {hmdSled = new XGLTransformer(); return hmdSled; });
	hmdSled->SetName("HmdSled");
	CreateShape("shaders/specular", [&](){ hmdChild = new XGLSphere(0.1, 32); return hmdChild; });
	hmdChild->attributes.diffuseColor = XGLColors::cyan;
	hmdSled->AddChild(hmdChild);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); return shape; });
	shape->SetName("LeftHand");
	hmdSled->AddChild(shape);

	CreateShape("shaders/specular", [&]() { shape = new XGLSphere(0.05f, 16); return shape; });
	shape->SetName("RightHand");
	hmdSled->AddChild(shape);

	/*
	XInputMouseFunc worldCursorMouse = [&](int x, int y, int flags) {
		if (mt.IsTrackingRightButton()) {
			XGLWorldCoord *out = wc.Unproject(projector, x, y);

			// project the worldCursor ray onto the X/Y (Z=0) plane
			// TODO: Figure this out.  I found it on StackOverflow.
			float f = out[0].z / (out[1].z - out[0].z);
			float x2d = out[0].x - f * (out[1].x - out[0].x);
			float y2d = out[0].y - f * (out[1].y - out[0].y);
			float z2d = -0.002f;

			if (sphere) {
				glm::mat4 translate = glm::translate(glm::mat4(), glm::vec3(x2d, y2d, z2d));
				sphere->model = translate;
			}
		}
	};
	AddMouseFunc(worldCursorMouse);
	*/

	/*
	AddProportionalFunc("RightHandTrigger", [](float v) { xprintf("RightHandTrigger: %0.3f\n", v); });
	AddProportionalFunc("RightIndexTrigger", [](float v) { xprintf("RightIndexTrigger: %0.3f\n", v); });
	AddProportionalFunc("RightThumbStick.x", [](float v) { xprintf("RightThumbStick.x: %0.3f\n", v); });
	AddProportionalFunc("RightThumbStick.y", [](float v) { xprintf("RightThumbStick.y: %0.3f\n", v); });
	AddProportionalFunc("LeftHandTrigger", [](float v) { xprintf("LeftHandTrigger: %0.3f\n", v); });
	AddProportionalFunc("LeftIndexTrigger", [](float v) { xprintf("LeftIndexTrigger: %0.3f\n", v); });
	*/
	AddProportionalFunc("LeftThumbStick.x", [](float v) {
		wcPos.x += v / 10.0f;
		glm::mat4 translate = glm::translate(glm::mat4(), wcPos);
		hmdSled->model = translate;
	});
	AddProportionalFunc("LeftThumbStick.y", [](float v) {
		wcPos.y += v / 10.0f;
		glm::mat4 translate = glm::translate(glm::mat4(), wcPos);
		hmdSled->model = translate;
	});
	AddProportionalFunc("LeftIndexTrigger", [](float v) {
		wcPos.z += v / 10.0f;
		glm::mat4 translate = glm::translate(glm::mat4(), wcPos);
		hmdSled->model = translate;
	});
	AddProportionalFunc("LeftHandTrigger", [](float v) {
		wcPos.z -= v / 10.0f;
		glm::mat4 translate = glm::translate(glm::mat4(), wcPos);
		hmdSled->model = translate;
	});
}