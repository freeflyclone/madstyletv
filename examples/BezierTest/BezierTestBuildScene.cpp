/**************************************************************
** BezierTestBuildScene.cpp
**
** Use GL_TRIANGLES as input to a quadratic Bezier shader.
** "bezier" shader implements a geometry shader that does
** piecewise interpolation of the bezier curve specified by
** a triangle, where the middle point is the "control" point.
**************************************************************/
#include "ExampleXGL.h"

class XGLNewTriangle : public XGLShape {
public:
	XGLNewTriangle() {
		SetName("XGLNewTriangle");
		v.push_back({ { -1, 0, 0 },    {0.0, 0.0}, {}, XGLColors::red });
		v.push_back({ { 0, 1.412, 0 }, {0.5, 0.0}, {}, {1, 0, 0, -1} });
		v.push_back({ { 1, 0, 0 },     {1.0, 1.0}, {}, XGLColors::red });
	}

	void Draw(){
		glPointSize(4.0f);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		GL_CHECK("glDrawArrays() failed");
	};
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLVertex triangle[3];

	AddShape("shaders/bezier", [&](){ shape = new XGLNewTriangle(); return shape; });
	shape->model = glm::scale(glm::mat4(), glm::vec3(8.0, 8.0, 8.0));
}
