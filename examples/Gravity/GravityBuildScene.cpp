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

const glm::vec3 g = { 0.0f, 0.0f, -9.80665f };
XGLSphere *shape;

void ExampleXGL::BuildScene() {
	extern bool initHmd;
	initHmd = false;
	AddShape("shaders/specular", [&](){ shape = new XGLSphere(0.5f, 64); return shape; });

	shape->p = { 0.0f, 0.0f, 10.0f };
	shape->m = 2.0f;
	shape->SetMatrix();

	// A bouncing ball simulation, with damping
	shape->SetAnimationFunction([&](float clock){
		XPhyBody *b = static_cast<XPhyBody *>(shape);

		float g = 9.8f;
		float dt = 0.01666f;

		float T = 0.5f * b->m * b->v.z * b->v.z;
		float V = b->m * (g*g) * b->p.z;

		float L = (T - V) * dt;
		xprintf("L: %0.4f\n", L);

		b->v.z -= g * dt;
		b->p.z += b->v.z;
		
		if (b->p.z < 0.0f) {
			b->p.z *= -0.9f;
			b->v.z *= -0.9f;
		}

		// set model matrix (affect the visuals) with the result
		b->SetMatrix();
	});
}
