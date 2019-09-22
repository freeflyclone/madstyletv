/**************************************************************
** BezierTestBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"

class XGLNewTriangle : public XGLShape {
public:
	XGLNewTriangle() {
		SetName("XGLNewTriangle");
		v.push_back({ { -1, 0, 0 }, {}, {}, XGLColors::red });
		v.push_back({ { 1, 0, 0 }, {}, {}, XGLColors::red });
		v.push_back({ { 0, 1.412, 0 }, {}, {}, {0, 0, 1, -1} });
	}

	void Draw(){
		glPointSize(4.0f);
		glDrawArrays(GL_POINTS, 0, 3);
		GL_CHECK("glDrawArrays() failed");
	};
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLVertex triangle[3];

	AddShape("shaders/bezier", [&](){ shape = new XGLNewTriangle(); return shape; });

	for (int i = 0; i < 3; i++)
		triangle[i] = shape->v[i].v;

	// shader must be coded such that "triangle" uniform doesn't get optimized out by the shader compiler
	// ie: it has to be used somehow.
	GLint location = glGetUniformLocation(shape->shader->programId, "triangle");
	GL_CHECK("glGetUniformLocation() failed");
	if (location != -1) {
		xprintf("Found uniform \"triangle\"\n");
		glUniform3fv(location, 3, glm::value_ptr(triangle[0]));
		GL_CHECK("glUniform3fv() failed");
	}
}
