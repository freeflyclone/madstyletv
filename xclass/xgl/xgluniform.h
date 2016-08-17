/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL uniform object classes:
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
****************************************************************************/
#ifndef XGLUNIFORM_H
#define XGLUNIFORM_H

class XGLUniformf {
public:
    XGLUniformf(GLint p, std::string n, glm::vec3 v);
    
    std::string name;
    GLint location;
    GLfloat *value;
};

#endif