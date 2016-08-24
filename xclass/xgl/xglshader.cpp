#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#ifndef _WIN32
//#include <unistd.h>
#endif

#include "xgl.h"

XGLShaderComponent::XGLShaderComponent() : mSourceSize(0) {
}

XGLShaderComponent::~XGLShaderComponent() {
}

void XGLShaderComponent::TextFileRead() {
    std::ifstream ifd(mFileName.c_str(), std::ifstream::in);

    if (ifd){
        std::ostringstream contents;
        contents << ifd.rdbuf();
        ifd.close();
        mSourceString = contents.str();
        return;
    }
    throwXGLException("Couldn't find file: '" + 
        mFileName + 
        "'. \n\nThe file could not be found relative to:\n\n"+
        currentWorkingDir
    );
}

bool XGLShaderComponent::Compile(std::string name, GLuint type)
{
    char *sourceText;
    mFileName = name + (const char *)((type==GL_VERTEX_SHADER)?".vert":".frag");
	TextFileRead();
	
    sourceText = (char *)mSourceString.c_str();
 
	mShader = glCreateShader(type);
    GL_CHECK("glCreateShader() failed");
	glShaderSource(mShader, 1, (const GLchar **)&sourceText, NULL);
    GL_CHECK("glShaderSource() failed");
	glCompileShader(mShader);
    GL_CHECK("glCompileShader() failed");
    InfoLog(mFileName);
    return true;
}

void XGLShaderComponent::InfoLog(std::string fileName) {
	char infoLog[8192];
	GLint infoLogLength = sizeof(infoLog);
	GLint status = 0;
    int nWritten;

	glGetShaderiv(mShader, GL_COMPILE_STATUS, &status);
	if( status == GL_FALSE ) {
		xprintf("File: %s: Shader did NOT compile.\n",fileName.c_str());
		glGetShaderInfoLog(mShader, infoLogLength, &nWritten, infoLog);
		xprintf("%s\n\n", infoLog);
	}
}

XGLShader::XGLShader() : shader(-1) {} 
XGLShader::~XGLShader() { }

bool XGLShader::Compile(std::string name) {
    GLint status;

    if (!mVShader.Compile(name, GL_VERTEX_SHADER))
        throwXGLException("Compiling vertex shader failed.");
    if (!mFShader.Compile(name, GL_FRAGMENT_SHADER))
        throwXGLException("Compiling fragment shader failed.");

    shader = glCreateProgram();
    GL_CHECK("glCreateProgram() failed");

	glAttachShader(shader, mVShader.mShader);
    GL_CHECK("glAttachShader(VERTEX) failed");
    glAttachShader(shader, mFShader.mShader);
    GL_CHECK("glAttachShader(FRAGMENT) failed");
    glBindAttribLocation(shader, 0, "in_Position");
    GL_CHECK("glBindAttribLocation(\"in_Position\") failed");
	glBindAttribLocation(shader, 1, "in_TexCoord");
	GL_CHECK("glBindAttribLocation(\"in_TexCoord\") failed");
	glBindAttribLocation(shader, 2, "in_Normal");
    GL_CHECK("glBindAttribLocation(\"in_Normal\") failed");
    glBindAttribLocation(shader, 3, "in_Color");
    GL_CHECK("glBindAttribLocation(\"in_Color\") failed");

	glLinkProgram(shader);
    GL_CHECK("glLinkProgram() failed");
    glValidateProgram(shader);
    GL_CHECK("glValidateProgram() failed");
    
    glGetProgramiv(shader, GL_LINK_STATUS, &status);
    GL_CHECK("glGetProgramiv() failed");
    if (status != GL_TRUE)
        GL_CHECK("compile failed in the link stage");
    glGetProgramiv(shader, GL_VALIDATE_STATUS, &status);
    GL_CHECK("glGetProgramiv() failed");
    glGetProgramiv(shader, GL_ATTACHED_SHADERS, &status);
    GL_CHECK("glGetProgramiv() failed");
    glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &status);
    GL_CHECK("glGetProgramiv() failed");

	GLuint idx = glGetUniformBlockIndex(shader, "ShaderMatrixData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx == GL_INVALID_INDEX) {
		xprintf("ShaderMatrixData not found in '%s'\n", name.c_str());
	}
	else {
		// assign to uniform block 0
		glUniformBlockBinding(shader, idx, 0);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	idx = glGetUniformBlockIndex(shader, "LightData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx == GL_INVALID_INDEX) {
		xprintf("LightData not found in '%s'\n", name.c_str());
	}
	else {
		// assign to uniform block 1
		glUniformBlockBinding(shader, idx, 1);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	glUseProgram(shader);

	XGLLights lights = XGL::getInstance()->lights;
	XGLLight light = lights.back();

	glUniform3fv(glGetUniformLocation(shader, "light.position"), 1, (GLfloat*)glm::value_ptr(light.position));
	GL_CHECK("glUniform3fv() failed");

	glUniform3fv(glGetUniformLocation(shader, "light.intensities"), 1, (GLfloat*)glm::value_ptr(light.diffuse));
	GL_CHECK("glUniform3fv() failed");

	glUniform1f(glGetUniformLocation(shader, "light.attenuation"), light.attenuation);
	GL_CHECK("glUniform3fv() failed");

	glUniform1f(glGetUniformLocation(shader, "light.ambientCoefficient"), light.ambientCoefficient);
	GL_CHECK("glUniform3fv() failed");

	glUniform3fv(glGetUniformLocation(shader, "materialSpecularColor"), 1, (GLfloat*)glm::value_ptr(white));
	GL_CHECK("glUniform3fv() failed");

	glUniform1f(glGetUniformLocation(shader, "materialShininess"), 100.0f);
	GL_CHECK("glUniform1f() failed");

	//DebugPrintf("XGLShader::Compile(%s) shader #: %d\n", name.c_str(), shader);
    return true;
}

void XGLShader::InfoLog() {
	int infoLogLength =0;
	int nWritten;
	char infoLog[8192];

	glValidateProgram(shader);
    GL_CHECK("glValidateProgram() failed");
	glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
    GL_CHECK("glGetProgramiv() failed");
	if( infoLogLength ) {
		glGetProgramInfoLog(shader, infoLogLength, &nWritten, infoLog);
		//xprintf("%s\n\n", infoLog);
	}
}

void XGLShader::Use() {
    GLint currentProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&currentProgram);
    GL_CHECK("glGetIntegerv() failed");
    if( shader != currentProgram) {
        glUseProgram(shader);
        GL_CHECK("glUseProgram(shader) failed");
    }
}

void XGLShader::UnUse() {
	glUseProgram(0);
    GL_CHECK("glUseProgram(0) failed");
}

void XGLShader::Reshape(int w, int h){
	xprintf("XGLShader::Reshape(%d,%d) for '%s'\n", w, h, shaderName.c_str());
}
