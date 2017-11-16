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

void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/specular", [&](){ shape = new XGLSphere(0.5f, 64); return shape; });
	shape->p = { 0.0f, 0.0f, 10.0f };
	shape->SetMatrix();
}
