/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of 3D lighting object classes:
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
****************************************************************************/
#ifndef XGLLIGHT_H
#define XGLLIGHT_H

// defined as struct, so it will be passable directly to OpenGL shader(s)
// via a glUniform() call.
struct XGLLight {
    XGLVertex position;
    XGLColor diffuse;
};

#endif