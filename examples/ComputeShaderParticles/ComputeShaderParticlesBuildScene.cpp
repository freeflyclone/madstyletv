/**************************************************************
** ComputeShaderTestBuildScene.cpp
**
** The usual ground plane and controls, along with an example
** of creating a compute shader and rendering its output as
** a texture map.
**************************************************************/
#include "ExampleXGL.h"
#include "uniforms.h"

namespace {
	float roll = 0.1f;
};

GLuint createNoiseTexture4f3D(int w = 16, int h = 16, int d = 16, GLint internalFormat = GL_RGBA8_SNORM)
{
	uint8_t *data = new uint8_t[w*h*d * 4];
	uint8_t *ptr = data;
	for (int z = 0; z<d; z++) {
		for (int y = 0; y<h; y++) {
			for (int x = 0; x<w; x++) {
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
				*ptr++ = rand() & 0xff;
			}
		}
	}

	GLuint tex;
	glGenTextures(1, &tex);
	GL_CHECK("glGenTextues() failed");
	glBindTexture(GL_TEXTURE_3D, tex);
	GL_CHECK("failed");

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	GL_CHECK("failed");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	GL_CHECK("failed");

	//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, w, h, d, 0, GL_RGBA, GL_BYTE, data);
	GL_CHECK("failed");

	delete[] data;
	return tex;
}

ShaderParams shaderParams;
GLuint mUBO, mVBO;
float vtx_data[] = { 0.0f, 0.0f, 0.0f };
const static int mNumParticles = 1 << 20;
GLuint m_noiseTex;
GLuint posSSBO, velSSBO;

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	glm::mat4 translate, scale, rotate;

	AddShape("shaders/000-simple", [&]() { shape = new XGLPointCloud(mNumParticles); return shape; });

	// create the compute shader program object
	XGLShader *computeShader = new XGLShader("shaders/particles.comp");
	computeShader->CompileCompute(pathToAssets + "/shaders/particles.comp");

	glGenBuffers(1, &posSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, posSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, mNumParticles * sizeof(XGLVertex), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	GL_CHECK("position SSBO setup failed");

	glGenBuffers(1, &velSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, velSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, mNumParticles * sizeof(XGLVertex), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	GL_CHECK("velocity SSBO setup failed");

	m_noiseTex = createNoiseTexture4f3D();

	if (true)
	AddPreRenderFunction([computeShader](float clock) {
		glActiveTexture(GL_TEXTURE0);
		GL_CHECK("glActiveTexture failed");

		glBindBufferBase(GL_UNIFORM_BUFFER, 1, mUBO);
		GL_CHECK("glBindBufferBase failed");
		glBindBuffer(GL_UNIFORM_BUFFER, mUBO);
		GL_CHECK("glBindBuffer() failed");
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(ShaderParams), &shaderParams);
		GL_CHECK("glBufferSubData() failed");

		glBindTexture(GL_TEXTURE_3D, m_noiseTex);
		GL_CHECK("glBindTexture failed");

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, posSSBO);
		GL_CHECK("glBindBufferBase failed");
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, velSSBO);
		GL_CHECK("glBindBufferBase failed");

		// Invoke the compute shader to integrate the particles
		glBindProgramPipeline(computeShader->programId);
		GL_CHECK("glBindProgramPipeline failed");

		glDispatchCompute(mNumParticles / WORK_GROUP_SIZE, 1, 1);
		GL_CHECK("glDispatchCompute failed");

		// We need to block here on compute completion to ensure that the
		// computation is done before we render
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		GL_CHECK("glMemoryBarrier failed");

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
		GL_CHECK("glBindBufferBase failed");
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
		GL_CHECK("glBindBufferBase failed");

		glBindProgramPipeline(0);
		GL_CHECK("glBindProgramPipeline(0) failed");
	});

	//create ubo and initialize it with the structure data
	glGenBuffers(1, &mUBO);
	glBindBuffer(GL_UNIFORM_BUFFER, mUBO);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(ShaderParams), &shaderParams, GL_STREAM_DRAW);
	GL_CHECK("Generating UBO objects failed");

	glGenBuffers(1, &mVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vtx_data), vtx_data, GL_STATIC_DRAW);
	GL_CHECK("Generating VBO objects failed");
}
