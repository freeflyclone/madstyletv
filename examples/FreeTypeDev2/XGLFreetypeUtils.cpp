#include "XGLFreetypeUtils.h"

XGLFreetypeGrid::XGLFreetypeGrid(XGL* pxgl, XGLVertexList vList, FT::BoundingBox bb) : pXgl(pxgl) {
	v.push_back({ { bb.ul.x, bb.ul.y, 0 }, {}, {}, XGLColors::cyan });
	v.push_back({ { bb.lr.x, bb.ul.y, 0 }, {}, {}, XGLColors::cyan });

	v.push_back({ { bb.lr.x, bb.ul.y, 0 }, {}, {}, XGLColors::cyan });
	v.push_back({ { bb.lr.x, bb.lr.y, 0 }, {}, {}, XGLColors::cyan });

	v.push_back({ { bb.lr.x, bb.lr.y, 0 }, {}, {}, XGLColors::cyan });
	v.push_back({ { bb.ul.x, bb.lr.y, 0 }, {}, {}, XGLColors::cyan });

	v.push_back({ { bb.ul.x, bb.lr.y, 0 }, {}, {}, XGLColors::cyan });
	v.push_back({ { bb.ul.x, bb.ul.y, 0 }, {}, {}, XGLColors::cyan });

	for (auto vrtx : vList) {
		float x = vrtx.v.x;
		float y = vrtx.v.y;

		v.push_back({ { bb.ul.x, y, 0 }, {}, {}, XGLColors::cyan });
		v.push_back({ { bb.lr.x, y, 0 }, {}, {}, XGLColors::cyan });

		v.push_back({ { x, bb.ul.y, 0 }, {}, {}, XGLColors::cyan });
		v.push_back({ { x, bb.lr.y, 0 }, {}, {}, XGLColors::cyan });
	}
}

void XGLFreetypeGrid::Draw() {
	if (v.size() && drawGrid) {
		glDrawArrays(GL_LINES, 0, v.size());
		GL_CHECK("glDrawArrays() failed");
	}
}

XGLFreetypeProbe::XGLFreetypeProbe(XGL* pxgl) : pXgl(pxgl) {
	xprintf("%s()\n", __FUNCTION__);

	pXgl->AddShape("shaders/specular", [&]() { cube = new XGLCube(); return cube; });
	pXgl->AddShape("shaders/specular", [&]() { cubeX = new XGLCube(); return cubeX; });
	pXgl->AddShape("shaders/specular", [&]() { cubeY = new XGLCube(); return cubeY; });
}

void XGLFreetypeProbe::Move(XGLVertex v, FT::BoundingBox bb) {
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.0375, 0.0375, 0.0375));
	glm::mat4 translate = glm::translate(glm::mat4(), v);
	cube->model = translate * scale;

	translate = glm::translate(glm::mat4(), glm::vec3(bb.lr.x, v.y, 0.0));
	cubeX->model = translate * scale;

	translate = glm::translate(glm::mat4(), glm::vec3(v.x, bb.ul.y, 0.0));
	cubeY->model = translate * scale;
}
