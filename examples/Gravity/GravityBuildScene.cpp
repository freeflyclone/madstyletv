/**************************************************************
** GravityBuildScene.cpp
**
** Simulate gravity: a ground plane and a bouncing ball.
**
** To learn about basic physics math, it helps to have visual
** representations of what the math actually does.
**
** Here, we'll look at Kinetic Energy (KE), elastic collisions,
** and friction.
**************************************************************/
#include "ExampleXGL.h"
#include "xphybody.h"

XGLSphere *particle;
XGLSphere *anchor;

void ExampleXGL::BuildScene() {
	extern bool initHmd;
	initHmd = false;

	AddShape("shaders/specular", [&]() { anchor = new XGLSphere(1.0, 64); return anchor; });
	AddShape("shaders/specular", [&](){ particle = new XGLSphere(0.5f, 64); return particle; });

	// setup physics initial conditions for "anchor" (aka attractor)
	anchor->attributes.diffuseColor = XGLColors::yellow;
	anchor->m = 20.0f;

	// setup initial physics state for "particle"
	XPhyBody& b = *(XPhyBody*)particle;
	XPhyPoint initialPosition{ 0, 5.0, 0 };
	XPhyVelocity initialVelocity{ 0.1, 0.1, 0.1 };
	XPhyMass initialMass{ 2.0 };
	XPhySpeed initialSpeed{ 0.05 };

	b = { initialMass, initialSpeed, initialPosition, initialVelocity };

	// set model matrix (affect the visuals) with the result
	b.SetMatrix();

	// gravitational attraction between "anchor" and "particle"
	// (at least that's the intent, not there yet)
	particle->SetAnimationFunction([&](float clock){
		XPhyBody& b = *static_cast<XPhyBody*>(particle);
		XPhyBody& a = *static_cast<XPhyBody*>(anchor);

		XPhyDirection direction = glm::normalize(a.p - b.p);
		XPhyDirection stepDir = direction * b.s;

		b.v += stepDir;
		b.p += b.v;

		b.SetMatrix();
	});

	XInputKeyFunc resetBall = [&](int key, int flags) {
		const bool isDown = (flags & 0x8000) == 0;
		const bool isRepeat = (flags & 0x4000) != 0;
		static bool wireFrameMode = false;

		if (isDown && !isRepeat) {
			XPhyBody *b = static_cast<XPhyBody *>(particle);
			b->p = { 0, 0, 10 };
			b->v = { 0, 0, 0 };
		}
	};

	AddKeyFunc('R', resetBall);
	AddKeyFunc('r', resetBall);
}
