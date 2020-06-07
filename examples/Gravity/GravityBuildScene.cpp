/**************************************************************
** GravityBuildScene.cpp
**
** Simulate gravity: an adventure in n-body simulation math.
**
** My knowledge of math is weaker than I'd like.
**
** I'm curious how physics simulations actually work,
** and implementing my own (however imperfect) will require 
** me to advance my math skills, and that will hopefully stave 
** off mental decay as I age.
**************************************************************/
#include "ExampleXGL.h"
#include "xphybody.h"

XGLSphere *p1, *p2;

XPhyPoint initialPosition{ 10, 0, 0 };
XPhyVelocity initialVelocity{ 0.0, 0.0, 0.0 };
XPhyMass initialMass{ 2.0 };
XPhySpeed initialSpeed{ 0.0 };
XPhyForce force;

XPhyMagnitude g{ 9.8 };
XPhyMagnitude dt{ 0.01 };
bool isRunning{ true };

bool ctlWindow{ true };

void ExampleXGL::BuildScene() {
	extern bool initHmd;
	initHmd = false;

	AddShape("shaders/specular", [&]() { p1 = new XGLSphere(1.0, 64); return p1; });
	AddShape("shaders/specular", [&](){ p2 = new XGLSphere(0.5f, 64); return p2; });

	// setup initial initial state for "p1"
	XPhyBody& b1 = *(XPhyBody*)p1;
	b1 = { 20, initialSpeed, {0,0,0}, {0,0,0} };
	b1.SetMatrix();

	// setup initial physics state for "p2"
	XPhyBody& b2 = *(XPhyBody*)p2;
	b2 = { initialMass, initialSpeed, initialPosition, initialVelocity };
	b2.SetMatrix();

	// gravitational attraction between p1 and p2
	p2->SetAnimationFunction([&](float clock){
		XPhyBody& b1 = *static_cast<XPhyBody*>(p1);
		XPhyBody& b2 = *static_cast<XPhyBody*>(p2);
		
		if (isRunning) {
			XPhyDirection dir = glm::normalize(b1.p - b2.p);
			float distance = glm::length(b1.p - b2.p);

			if (distance == 0.0f)
				distance = 0.0000000001f;

			force = { dir, g * (b1.m * b2.m / pow(distance, 2)) };

			b2.v += (force.d * dt);
			b2.p += b2.v;

			b2.SetMatrix();
		}
	});

	XInputKeyFunc resetBall = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool wireFrameMode = false;

		if (isDown && !isRepeat) {
			XPhyBody& b = *(XPhyBody*)p2;
			b = { initialMass, initialSpeed, initialPosition, initialVelocity };
		}
	};

	AddKeyFunc('R', resetBall);
	AddKeyFunc('r', resetBall);
	menuFunctions.push_back(([&]() {
		if (ImGui::Begin("Gravity Controls", &ctlWindow))
		{
			int changes{ 0 };
			changes += ImGui::Checkbox("Running", &isRunning);

			ImGui::SameLine();

			changes += ImGui::SliderFloat("dt", &dt, 0.0f, 1.0f, "%0.4f");
			changes += ImGui::SliderFloat("initial x", &initialPosition.x, -100.0f, 100.0f, "%0.4f");
			changes += ImGui::SliderFloat("initial y", &initialPosition.y, -100.0f, 100.0f, "%0.4f");
			changes += ImGui::SliderFloat("initial z", &initialPosition.z, -100.0f, 100.0f, "%0.4f");

			ImGui::SliderFloat("p2.p.x", &p2->p.x, -100.0f, 100.0f, "%0.4f");
			ImGui::SliderFloat("p2.p.y", &p2->p.y, -100.0f, 100.0f, "%0.4f");
			ImGui::SliderFloat("p2.p.z", &p2->p.z, -100.0f, 100.0f, "%0.4f");

			ImGui::Text("Force: %0.4f", force.m);

			if (!isRunning && changes) {
				XPhyBody& b = *(XPhyBody*)p2;
				b = { initialMass, initialSpeed, initialPosition, initialVelocity };
				b.SetMatrix();
			}
		}
		ImGui::End();
	}));
}
