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
	XGLMaterial();

	void Bind(GLuint program);

	XGLMaterialAttributes m;
};

#endif