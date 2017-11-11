#include "ExampleXGL.h"


class XGLParticleSystem : public XGLPointCloud {
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

	VertexList verts;
	GLuint vbo,vao;
};

