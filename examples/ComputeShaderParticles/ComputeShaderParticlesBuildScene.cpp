/**************************************************************
** ComputeShaderParticlesBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader particle system.
**************************************************************/
#include "ExampleXGL.h"
#include "Particles.h"

const static int nParticles = 1024*768;

XGLParticleSystem *pParticles;

extern bool initHmd;

void ExampleXGL::BuildScene() {
	AddShape("shaders/000-simple", [&]() { pParticles = new XGLParticleSystem(nParticles); return pParticles; });

	if (true)
		AddPreRenderFunction(pParticles->invokeComputeShader);

	initHmd = false;

	preferredSwapInterval = 0;
}
