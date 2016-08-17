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

const XGLColor black = { 0,0,0,1 };
const XGLColor white = { 1,1,1,1 };
const XGLColor red = { 1,0,0,1 };
const XGLColor green = { 0,1,0,1 };
const XGLColor blue = { 0,0,1,1 };
const XGLColor yellow = { 1,1,0,1 };

class XGLMaterial : public XGLObject {
public:
	XGLMaterial() : XGLObject("XGLMaterial"),
		ambientColor(white),
		diffuseColor(white),
		specularColor(white),
		shininess(1)
	{};

	void Bind(GLuint program) {
		glUniform4fv(glGetUniformLocation(program, "ambientColor"), 1, glm::value_ptr(ambientColor));
		GL_CHECK("glUniform4fv() failed");
		glUniform4fv(glGetUniformLocation(program, "diffuseColor"), 1, glm::value_ptr(diffuseColor));
		GL_CHECK("glUniform4fv() failed");
	}
	XGLColor ambientColor;
	XGLColor diffuseColor;
	XGLColor specularColor;
	glm::float32 shininess;
};

#endif