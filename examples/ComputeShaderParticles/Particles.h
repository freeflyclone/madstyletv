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
		glm::vec4 tex;
		glm::vec4 norm;
		glm::vec4 color;
	};

	typedef std::vector<VertexAttributes> VertexList;

	XGLParticleSystem(int n = 0);
	virtual void Draw();
	AnimationFn invokeComputeShader;

	VertexList verts;
	GLuint vbo,vao;
	XGLShader *computeShader;
};

