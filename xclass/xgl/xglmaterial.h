/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of 3D Materials object classes:
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
****************************************************************************/
#ifndef XGLMATERIAL_H
#define XGLMATERIAL_H

#include "xglprimitives.h"

#define Z 0.0001

const XGLColor black =   { Z, Z, Z, 1 };
const XGLColor white =   { 1, 1, 1, 1 };
const XGLColor red =     { 1, Z, Z, 1 };
const XGLColor green =   { Z, 1, Z, 1 };
const XGLColor blue =    { Z, Z, 1, 1 };
const XGLColor yellow =  { 1, 1, Z, 1 };
const XGLColor cyan =    { Z, 1, 1, 1 };
const XGLColor magenta = { 1, Z, 1, 1 };

class XGLMaterial {
public:
	XGLMaterial() {
		m.ambientColor = white;
		m.diffuseColor = white;
		m.specularColor = white;
		m.shininess = 120.0f;
	};

	void Bind(GLuint program) {
		glUniform4fv(glGetUniformLocation(program, "ambientColor"), 1, glm::value_ptr(m.ambientColor));
		GL_CHECK("glUniform4fv() failed");
		glUniform4fv(glGetUniformLocation(program, "diffuseColor"), 1, glm::value_ptr(m.diffuseColor));
		GL_CHECK("glUniform4fv() failed");
	}

	XGLMaterialAttributes m;
};

#endif