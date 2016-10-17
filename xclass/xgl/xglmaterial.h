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

typedef struct _XGLMaterialAttributes
{
	XGLColor ambientColor;
	XGLColor diffuseColor;
	XGLColor specularColor;
	GLfloat shininess;
} XGLMaterialAttributes;

typedef struct _XGLMaterialLocations
{
	GLint ambientLocation;
	GLint diffuseLocation;
	GLint specularLocation;
	GLint shininessLocation;
} XGLMaterialLocations;

class XGLMaterial {
public:
	XGLMaterial();

	void Bind(GLuint program);

	XGLMaterialAttributes attributes;
	XGLMaterialLocations l;
};

#endif