#include "XGLFreetypeUtils.h"

XGLFreetypeGrid::XGLFreetypeGrid(XGL* pxgl, XGLVertexList vList, FT::BoundingBox b) : pXgl(pxgl), bb(b) {
	v.push_back({ { bb.ul.x, bb.ul.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
	v.push_back({ { bb.lr.x, bb.ul.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });

	v.push_back({ { bb.lr.x, bb.ul.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
	v.push_back({ { bb.lr.x, bb.lr.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });

	v.push_back({ { bb.lr.x, bb.lr.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
	v.push_back({ { bb.ul.x, bb.lr.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });

	v.push_back({ { bb.ul.x, bb.lr.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
	v.push_back({ { bb.ul.x, bb.ul.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });

	for (auto vrtx : vList) {
		float x = vrtx.v.x;
		float y = vrtx.v.y;

		v.push_back({ { bb.ul.x, y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
		v.push_back({ { bb.lr.x, y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });

		v.push_back({ { x, bb.ul.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
		v.push_back({ { x, bb.lr.y, 0 }, {}, {}, {0.5, 0.5, 0.5, 1.0} });
	}
}

void XGLFreetypeGrid::Move(int i) {
	if (i<0 || i>v.size() - 4)
		return;

	idx = i;
}

void XGLFreetypeGrid::Draw() {
	if (!draw || v.size() < 8)
		return;

	if (drawBorder) {
		glDrawArrays(GL_LINES, 0, 8);
		GL_CHECK("glDrawArrays() failed");
	}

	if (drawUpTo) {
		glDrawArrays(GL_LINES, 8, idx * 4);
		GL_CHECK("glDrawArrays() failed");
	}

	if (drawFromHere) {
		glDrawArrays(GL_LINES, 8 + ((idx + 1) * 4), v.size() - ((idx + 1) * 4) - 8);
		GL_CHECK("glDrawArrays() failed");
	}
}

XGLFreetypeProbe::XGLFreetypeProbe(XGL* pxgl) : pXgl(pxgl) {
	xprintf("%s()\n", __FUNCTION__);

	pXgl->AddShape("shaders/specular", [&]() { sphere = new XGLSphere(1.0, 32); return sphere; });
	pXgl->AddShape("shaders/specular", [&]() { sphereX = new XGLSphere(1.0, 32); return sphereX; });
	pXgl->AddShape("shaders/specular", [&]() { sphereY = new XGLSphere(1.0, 32); return sphereY; });
}

void XGLFreetypeProbe::Move(XGLVertex v, FT::BoundingBox bb) {
	glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(0.0375, 0.0375, 0.0375));
	glm::mat4 translate = glm::translate(glm::mat4(), v);
	sphere->model = translate * scale;

	translate = glm::translate(glm::mat4(), glm::vec3(bb.lr.x, v.y, 0.0));
	sphereX->model = translate * scale;

	translate = glm::translate(glm::mat4(), glm::vec3(v.x, bb.ul.y, 0.0));
	sphereY->model = translate * scale;
}

XGLFreetypeCrosshair::XGLFreetypeCrosshair(XGL* pxgl, XGLVertexList vList, FT::BoundingBox b) : pXgl(pxgl), bb(b) {
	for (auto vrtx : vList) {
		float x = vrtx.v.x;
		float y = vrtx.v.y;

		v.push_back({ { bb.ul.x, y, 0 }, {}, {}, XGLColors::magenta });
		v.push_back({ { bb.lr.x, y, 0 }, {}, {}, XGLColors::magenta });

		v.push_back({ { x, bb.ul.y, 0 }, {}, {}, XGLColors::magenta });
		v.push_back({ { x, bb.lr.y, 0 }, {}, {}, XGLColors::magenta });
	}
}

void XGLFreetypeCrosshair::Move(int i) {
	if (i<0 || i>v.size() - 4)
		return;

	idx = i;
}

void XGLFreetypeCrosshair::Draw() {
	if (v.size() && draw) {
		glDrawArrays(GL_LINES, idx*4, 4);
		GL_CHECK("glDrawArrays() failed");
	}
}

