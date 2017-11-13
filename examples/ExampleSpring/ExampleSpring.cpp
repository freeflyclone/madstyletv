/**************************************************************
** Example03: demonstrate instantiation of a "ground"
** plane and multiple toruses with a lighting shader,
** transformations, animation functions, and child object
** chains
**************************************************************/
#include "ExampleXGL.h"

// this needs to be file scope at least.  Local (to ::BuildScene) doesn't work

// resting length;
float length{ 10.0f };
float k{ 0.1f };
float f{};
float m{ 4 };
float v{};

float initialLength{20.0f};

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

	ball2->SetAnimationFunction([ball1, ball2](float clock) {
		static float oldClock = 0.0f;
		// animation functions are being called twice per loop for unknown reasons. FIX IT!!
		if (oldClock != clock) {
			oldClock = clock;

			// ball1 doesn't move, ball2 does
			{
				// get current displacement of ball2 from resting length of spring 
				float d = ball2->p.x - (ball1->p.x + length);

				// calculate spring force: displacement * mass
				f = k * d * m;

				// add friction force (damping)
				f -= v*0.01;

				// accumulate calculated force in velocity
				v += f;

				// 
				ball2->p.x -= v * 0.01;

				ball2->model = glm::translate(glm::mat4(), ball2->p);
			}
		}
	});
}