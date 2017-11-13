/**************************************************************
** ExampleSpring: Simple spring dynamics simulation
**************************************************************/
#include "ExampleXGL.h"

class Spring {
public:
	Spring(XGLVertex p1, XGLVertex p2) : p1(p1), p2(p2) {};

	void Integrate() {
		// get current displacement of ball2 from resting length of spring 
		float d = p2.x - (p1.x + length);

		// calculate spring force: displacement * mass
		f = k * d * m;

		// add friction force (damping)
		f -= v * 0.01;

		// accumulate calculated force in velocity
		v += f;

		p2.x -= v * 0.01;
	}

	XGLVertex P2() { return p2; };

private:
	XGLVertex p1, p2;

	// resting length;
	float length { 10.0f };
	float k{ 0.1f };
	float f{};
	float m{ 4 };
	float v{};
};

Spring *spring;

void ExampleXGL::BuildScene() {
	XGLSphere *ball1, *ball2;

	AddShape("shaders/specular", [&](){ ball1 = new XGLSphere(1.0f, 32); return ball1; });
	ball1->SetName("Ball");
	ball1->model = glm::translate(glm::mat4(), glm::vec3(-10, 0, 0));
	ball1->p = { -10.0f, 0.0f, 0.0f };

	AddShape("shaders/specular", [&](){ ball2 = new XGLSphere(1.0f, 32); return ball2; });
	ball2->SetName("Ball");
	ball2->model = glm::translate(glm::mat4(), glm::vec3(10, 0, 0));
	ball2->p = { 10.0f, 0.0f, 0.0f };

	spring = new Spring(ball1->p, ball2->p);

	ball2->SetAnimationFunction([ball2](float clock) {
		spring->Integrate();
		ball2->model = glm::translate(glm::mat4(), spring->P2());
	});
}