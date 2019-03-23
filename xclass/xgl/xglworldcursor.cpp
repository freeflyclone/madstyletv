#include "xglworldcursor.h"

XGLWorldCursor::XGLWorldCursor(XGLShaderMatrixData *s) : nCoords(0), smd(s) {
}

void XGLWorldCursor::Event(XGLCamera c, XGLProjector p, XGLWorldCoord i){
}

XGLWorldCoord* XGLWorldCursor::Unproject(XGLProjector p, int x, int y) {
	glm::vec4 viewport = glm::vec4(0, 0, p.width, p.height);
	glm::mat4 view = smd->view;
	glm::mat4 proj = smd->projection;
	glm::vec3 pos;

	// the first point is the window mouse x,y projected to far Z plane
	pos = { x, p.height - y, 1 };
	out[1] = glm::unProject(pos, view, proj, viewport);

	// the second point is the center of the XGLCamera projected to near Z plane
	pos = { p.width / 2, p.height / 2, 0 };
	out[0] = glm::unProject(pos, view, proj, viewport);

	// return a ray (2 points) projected to world space.
	// (We'll let some other method decide what to do with it)
	nCoords = 2;

	return out;
}


XGLWorldCoord* XGLWorldCursor::Unproject(XGLCamera c, XGLProjector p, int x, int y) {
	glm::vec4 viewport = glm::vec4(0, 0, p.width, p.height);
	glm::mat4 view = smd->view;
	glm::mat4 proj = smd->projection;
	glm::vec3 pos;

	// the first point is the window mouse x,y projected to far Z plane
	pos = { x, p.height - y, 1 };
	out[0] = glm::unProject(pos, view, proj, viewport);

	// the second point is the center of the XGLCamera projected to near Z plane
	pos = { p.width / 2, p.height / 2, 0 };
	out[1] = glm::unProject(pos, view, proj, viewport);

	// return a ray (2 points) projected to world space.
	// (We'll let some other method decide what to do with it)
	nCoords = 2;

	return out;
}

void XGLWorldCursor::UnprojectViewFrustum(glm::mat4 view, glm::mat4 proj, glm::vec4 viewport) {
	XGLWorldCoord pos;
	float width = viewport[2];
	float height = viewport[3];

	pos = { 0, 0, 0.001f };
	out[2] = glm::unProject(pos, view, proj, viewport);
	pos = { 0, height, 0.001f };
	out[3] = glm::unProject(pos, view, proj, viewport);
	out[4] = glm::unProject(pos, view, proj, viewport);
	pos = { width, height, 0.001f };
	out[5] = glm::unProject(pos, view, proj, viewport);
	out[6] = glm::unProject(pos, view, proj, viewport);
	pos = { width, 0, 0.001f };
	out[7] = glm::unProject(pos, view, proj, viewport);
	out[8] = glm::unProject(pos, view, proj, viewport);
	out[9] = out[2];

	pos = { 0, 0, 0.9f };
	out[10] = glm::unProject(pos, view, proj, viewport);
	pos = { 0, height, 0.9f };
	out[11] = glm::unProject(pos, view, proj, viewport);
	out[12] = glm::unProject(pos, view, proj, viewport);
	pos = { width, height, 0.9f };
	out[13] = glm::unProject(pos, view, proj, viewport);
	out[14] = glm::unProject(pos, view, proj, viewport);
	pos = { width, 0, 0.9f };
	out[15] = glm::unProject(pos, view, proj, viewport);
	out[16] = glm::unProject(pos, view, proj, viewport);
	out[17] = out[10];

	out[18] = out[2];
	out[19] = out[10];
	out[20] = out[3];
	out[21] = out[11];
	out[22] = out[5];
	out[23] = out[13];
	out[24] = out[7];
	out[25] = out[16];

	nCoords += 24;
}