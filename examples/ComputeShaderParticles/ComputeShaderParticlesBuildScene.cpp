/**************************************************************
** ComputeShaderParticlesBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader particle system.
**************************************************************/
#include "ExampleXGL.h"
#include "Particles.h"

const static int numParticles = 4;// << 16;

XGLParticleSystem *pParticles;

void ExampleXGL::BuildScene() {
	AddShape("shaders/000-simple2", [&]() { pParticles = new XGLParticleSystem(numParticles); return pParticles; });

	// create the compute shader program object
	if (false) {
		XGLShader *computeShader = new XGLShader("shaders/particle-system");
		computeShader->CompileCompute(pathToAssets + "/shaders/particle-system");

		// and cause it to be "dispatched" in the preRender phase
		AddPreRenderFunction([computeShader](float clock) {
			glUseProgram(computeShader->programId);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pParticles->vbo);
			glDispatchCompute((GLuint)pParticles->verts.size(), 1, 1);
			glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
			GL_CHECK("Dispatch compute shader");
		});
	}
}
