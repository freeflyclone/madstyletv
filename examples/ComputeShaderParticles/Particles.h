/**
	XGLParticleSystem a GLSL compute shader based particle system.

	An example of sharing a VBO with a compute shader, so as to affect vertex positions via
	the compute shader code.
*/
#include "ExampleXGL.h"

class XGLParticleSystem : public XGLShape {
public:
	struct VertexAttributes {
		glm::vec4 pos;
		glm::vec4 vel;
		glm::vec4 norm;
		glm::vec4 color;
	};

	typedef std::vector<VertexAttributes> VertexList;

	XGLParticleSystem(int n = 0);
	virtual void Draw();
	GLuint CreateNoiseTexture4f3D(int w = 16, int h = 16, int d = 16, GLint internalFormat = GL_RGBA8_SNORM);

	AnimationFn invokeComputeShader;

	VertexList verts;
	GLuint vbo{ 0 }, vao{ 0 };
	GLuint tex{ 0 };
	XGLShader *computeShader{ nullptr };
	int numParticles{ 0 };
	int maxInvocations{ 0 };
	int sx{ 0 }, sy{ 0 }, sz{ 0 };
	int cx{ 0 }, cy{ 0 }, cz{ 0 };
};

