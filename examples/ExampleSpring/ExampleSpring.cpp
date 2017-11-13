/**************************************************************
** ExampleSpring: Simple spring dynamics simulation
**************************************************************/
#include "ExampleXGL.h"

extern bool initHmd;

class Spring {
public:
	Spring(XGLShape* s1, XGLShape* s2) : s1(s1), s2(s2) {};

	void Integrate() {
		// get normalize direction vectors between the 2 shapes
		d1 = glm::normalize(s2->p - s1->p);
		d2 = glm::normalize(s1->p - s2->p);

		// get current displacement between shapes 
		float displacement = glm::distance(s2->p,s1->p);

		// adjust displacement against length of spring
		displacement -= length;

		// calculate spring force: displacement * mass
		float force = k * displacement * m;

		// substrace friction force (damping)
		force -= velocity * friction;

		// accumulate force in velocity
		velocity += force;

		// move shape 2 toward shape 1
		s2->p += d2 * (velocity * timeStep);
		// move shape 1 toward shape 2
		s1->p += d1 * (velocity * timeStep);

		// update their model matrices from their positions
		s1->model = glm::translate(glm::mat4(), s1->p);
		s2->model = glm::translate(glm::mat4(), s2->p);
	}
private:
	XGLShape *s1, *s2;
	XGLVertex d1,d2;

	// resting length, spring coefficient, damping, mass of end points
	float length{ 10.0f }, k{ 0.1f }, friction{ 0.05f }, m{ 4 }, velocity{ 0.0f }, timeStep{ 1.0f / 60.0f };
};

Spring *spring1, *spring2, *spring3;
Spring *spring4, *spring5, *spring6;
XGLSphere *ball1, *ball2, *ball3, *ball4;
const float bump{ 2.0 };

void ExampleXGL::BuildScene() {
	initHmd = true;

	AddShape("shaders/specular", [&](){ ball1 = new XGLSphere(1.0f, 32); return ball1; });
	ball1->SetName("Ball");
	ball1->p = { -10.0f, 0.0f, 0.0f };
	ball1->attributes.diffuseColor = XGLColors::red;

	AddShape("shaders/specular", [&](){ ball2 = new XGLSphere(1.0f, 32); return ball2; });
	ball2->SetName("Ball");
	ball2->p = { 10.0f, 0.0f, 10.0f };
	ball2->attributes.diffuseColor = XGLColors::green;

	AddShape("shaders/specular", [&](){ ball3 = new XGLSphere(1.0f, 32); return ball3; });
	ball3->SetName("Ball");
	ball3->p = { 0.0f, -5.0f, 0.0f };
	ball3->attributes.diffuseColor = XGLColors::blue;

	AddShape("shaders/specular", [&](){ ball4 = new XGLSphere(1.0f, 32); return ball4; });
	ball4->SetName("Ball");
	ball4->p = { 0.0f, 0.0f, 10.0f };
	ball4->attributes.diffuseColor = XGLColors::magenta;

	spring1 = new Spring(ball1, ball2);
	spring2 = new Spring(ball2, ball3);
	spring3 = new Spring(ball3, ball1);

	spring4 = new Spring(ball1, ball4);
	spring5 = new Spring(ball2, ball4);
	spring6 = new Spring(ball3, ball4);

	AddPreRenderFunction([](float clock) {
		spring1->Integrate(); 
		spring2->Integrate();
		spring3->Integrate();
		spring4->Integrate();
		spring5->Integrate();
		spring6->Integrate();
	});

	AddKeyFunc(1, [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		
		if (isDown && !isRepeat)
			ball1->p.x += bump;
	});
	AddKeyFunc('L', [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		
		if (isDown && !isRepeat)
			ball1->p.x += bump;
	});

	AddKeyFunc(2, [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball1->p.x -= bump;
	});
	AddKeyFunc('H', [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball1->p.x -= bump;
	});

	AddKeyFunc(3, [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball4->p.z += bump;
	});
	AddKeyFunc('K', [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball4->p.z += bump;
	});

	AddKeyFunc(4, [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball4->p.z -= bump;
	});
	AddKeyFunc('J', [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;

		if (isDown && !isRepeat)
			ball4->p.z -= bump;
	});
}
