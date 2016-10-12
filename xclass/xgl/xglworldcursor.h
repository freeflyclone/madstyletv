/*
** XGLWorldCursor.h
**
** A generic 3D cursor for manipulating objects in virtual world space.  Form of the
** cursor is TBD.  The general model is a device in the real world (starting with a mouse)
** that a user manipulates, that has corresponding actions in the virtual space.
**
** Initial implementation will be based on a 2D positioning device (a mouse),
** which is then projected into the world.  However, modern VR vendors are creating
** manipulation controllers that are 3D positioning devices, so creating a
** data model with that in mind is part of the process.
**
** The idea of a cursor in computer science needs no explanation.  This is just
** an extension of the concept to 3 dimenstions.
*/
#pragma once

#include <xgl.h>
#include "xinput.h"

typedef glm::vec3 XGLWorldCoord;

class XGLWorldCursor {
public:
	XGLWorldCursor(XGLShaderMatrixData *s);

	// reserved: for future input devices that may emit world space coordinates
	void Event(XGLCamera c, XGLProjector p, XGLWorldCoord i);

	// For mouse events: x,y coordinates of mouse create a ray in world coordinates:
	// the first point is the ray projected on the far Z plane, the 2nd point is
	// the center of the XGLCamera at it's current location is world space.
	XGLWorldCoord *Unproject(XGLCamera c, XGLProjector p, int x, int y);
	XGLWorldCoord *Unproject(XGLProjector p, int x, int y);

	XGLWorldCoord *GetWorldCoord() {
		return out;
	}

	int GetCoordCount() {
		return nCoords;
	}

private:
	XGLShaderMatrixData *smd;
	XGLWorldCoord in;
	XGLWorldCoord out[26];
	int nCoords;

	// for debugging, draw a view frustum that reflects the current state of the camera & projector passed into Event()
	void UnprojectViewFrustum(glm::mat4 view, glm::mat4 proj, glm::vec4 viewport);
};