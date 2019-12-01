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
#include "xgllight.h"
#include "xglmaterial.h"

class XGLShaderComponent
{
public:
    XGLShaderComponent();
    ~XGLShaderComponent();

    bool TextFileRead();
    bool Compile(std::string name, GLuint type);
    bool InfoLog(std::string fileName);

    std::string mFileName;
    std::streamoff mSourceSize;
    std::string mSourceString;
    GLuint	mShader;
};

class XGLShader
{
public:
    XGLShader(std::string name);
    ~XGLShader();

    bool Compile(std::string name);
	bool CompileCompute(std::string name);
    void InfoLog();
    void Reshape(int w, int h);
    void Use() const;
    void UnUse() const;

	std::string Name() const ;
	GLint Attrib(std::string name);
	GLint Uniform(std::string name);
	void SetUniform(std::string name, GLint v);

	GLint programId;
	GLint modelUniformLocation;
	XGLMaterialUniformLocations materialLocations;

private:
	std::string shaderName;
    XGLShaderComponent mVShader, mFShader, mGShader;
	XGLShaderComponent mCShader;
};


#endif