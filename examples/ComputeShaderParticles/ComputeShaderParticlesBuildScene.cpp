/**************************************************************
** ComputeShaderParticlesBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader particle system.
**************************************************************/
#include "ExampleXGL.h"
#include "Particles.h"

const static int nParticles = 100000;

XGLParticleSystem *pParticles;

void ExampleXGL::BuildScene() {
	AddShape("shaders/000-simple2", [&]() { pParticles = new XGLParticleSystem(nParticles); return pParticles; });

	if (true)
		AddPreRenderFunction(pParticles->invokeComputeShader);
}
