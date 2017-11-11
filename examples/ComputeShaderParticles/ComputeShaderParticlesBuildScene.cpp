/**************************************************************
** ComputeShaderParticlesBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader particle system.
**************************************************************/
#include "ExampleXGL.h"
#include "Particles.h"

const static int numParticles = 1 << 10;

XGLParticleSystem *pParticles;

void ExampleXGL::BuildScene() {
	AddShape("shaders/000-simple2", [&]() { pParticles = new XGLParticleSystem(numParticles); return pParticles; });

	if (true)
		AddPreRenderFunction(pParticles->invokeComputeShader);
}
