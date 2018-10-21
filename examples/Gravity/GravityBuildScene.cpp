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
	shape->m = 1.0f;
	shape->SetMatrix();

	// A bouncing ball simulation, with damping
	shape->SetAnimationFunction([&](float clock){
		XPhyBody *pb = static_cast<XPhyBody *>(shape);

		if (pb->p.z <= shape->radius) {
			pb->v *= -0.6f;				// semi elastic bounce
			pb->p.z = shape->radius;	// fix overshoot	
		}
		else
			pb->v += (g / 600.0f);		// integrate velocity

		pb->p += pb->v;

		// set model matrix (affect the visuals) with the result
		pb->SetMatrix();
	});
}
