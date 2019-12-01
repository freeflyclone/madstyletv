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
		XGLColor neonYellow = { 0.8, 1.0, 0.00001, -1.0 };
		XGLColor madstyleRed = { 0.85, 0.0, 0.15, 1.0 };

		SetName("XGLNewTriangle");
		v.push_back({ { 0, 0, 0 }, { 0.0, 0.0 }, {}, madstyleRed });
		v.push_back({ { 0, 1, 0 }, { 0.5, 0.0 }, {}, madstyleRed });
		v.push_back({ { 1, 1, 0 }, { 1.0, 1.0 }, {}, madstyleRed });

		v.push_back({ { 0, 0, 0 }, { 0.0, 0.0 }, {}, neonYellow });
		v.push_back({ { 0, 1, 0 }, { 0.5, 0.0 }, {}, neonYellow });
		v.push_back({ { -1, 1, 0 }, { 1.0, 1.0 }, {}, neonYellow });

		v.push_back({ { -1, 1, 0 }, { 1.0, 1.0 }, {}, madstyleRed });
		v.push_back({ { 0, 1, 0 }, { 0.5, 0.0 }, {}, madstyleRed });
		v.push_back({ { 0, 2, 0 }, { 0.0, 0.0 }, {}, madstyleRed });

		v.push_back({ { 0, 2, 0 }, { 0.0, 0.0 }, {}, neonYellow });
		v.push_back({ { 0, 1, 0 }, { 0.5, 0.0 }, {}, neonYellow });
		v.push_back({ { 1, 1, 0 }, { 1.0, 1.0 }, {}, neonYellow });
	}

	void Draw(){
		glPointSize(4.0f);
		glDrawArrays(GL_TRIANGLES, 0, v.size());
		GL_CHECK("glDrawArrays() failed");
	};
};

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	XGLVertex triangle[3];

	AddShape("shaders/bezierPix", [&](){ shape = new XGLNewTriangle(); return shape; });
	shape->model = glm::scale(glm::mat4(), glm::vec3(10.0, 10.0, 1.0));
}
