/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL GLSL shader objects:
**
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
**
****************************************************************************/
#ifndef XGLSHADER_H
#define XGLSHADER_H

#include "xglprimitives.h"

class XGLShaderComponent
{
public:
    XGLShaderComponent();
    ~XGLShaderComponent();

    void TextFileRead();
    bool Compile(std::string name, GLuint type);
    void InfoLog(std::string fileName);

    std::string mFileName;
    std::streamoff mSourceSize;
    std::string mSourceString;
    GLuint	mShader;
};

class XGLShader
{
public:
    XGLShader();
    ~XGLShader();

    bool Compile(std::string name);
    void InfoLog();
    void Reshape(int w, int h);
    void Use();
    void UnUse();

    void AddUniform(std::string name);

    GLint shader;
//private:
    std::string shaderName;
    XGLShaderComponent mVShader, mFShader;
};


#endif