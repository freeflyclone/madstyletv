/**************************************************************
** ComputeShaderParticlesBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader particle system.
**************************************************************/
#include "ExampleXGL.h"
#include "Particles.h"

const static int numParticles = 4;// << 16;

struct VertexAttributes {
	glm::vec4 pos;
	glm::vec4 tex;
	glm::vec4 norm;
	glm::vec4 color;
};

VertexAttributes vrtx[numParticles];
XGLParticleSystem *pParticles;
GLuint vbo, vao;

void ExampleXGL::BuildScene() {
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 10.0);

	for (int i = 0; i < numParticles; i++) {
		vrtx[i].pos.x = dis(gen);
		vrtx[i].pos.y = dis(gen);
		vrtx[i].pos.z = dis(gen);
		vrtx[i].pos.w = 1.0f;
		vrtx[i].color = { 1.0, 1.0, 1.0, 1.0 };
	}

	AddShape("shaders/000-simple2", [&]() { pParticles = new XGLParticleSystem(numParticles); return pParticles; });

	glGenBuffers(1, &vbo);
	glGenVertexArrays(1, &vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)16);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)32);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexAttributes), (void *)48);

	glBufferData(GL_ARRAY_BUFFER, sizeof(vrtx), &vrtx, GL_STATIC_DRAW);

	GL_CHECK("Oops, something bad happened");
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	pParticles->drawFn = [&]() {
		glPointSize(4.0f);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		GL_CHECK("glBindBuffer() failed");

		glDrawArrays(GL_POINTS, 0, GLsizei(numParticles));
		GL_CHECK("glDrawArrays() failed");
	};

	// create the compute shader program object
	XGLShader *computeShader = new XGLShader("shaders/particle-system");
	computeShader->CompileCompute(pathToAssets + "/shaders/particle-system");

	// and cause it to be "dispatched" in the preRender phase
	if (true)
	AddPreRenderFunction([computeShader](float clock) {
		glUseProgram(computeShader->programId);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo);
		glDispatchCompute(numParticles, 1, 1);
		glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
		GL_CHECK("Dispatch compute shader");
	});
}
