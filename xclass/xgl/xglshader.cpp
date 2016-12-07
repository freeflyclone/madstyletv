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
	GLint sourceLength;
	std::string ext;

	if (type == GL_VERTEX_SHADER)
		ext.assign(".vert");
	else if (type == GL_FRAGMENT_SHADER)
		ext.assign(".frag");
	else
		ext.assign("");

	mFileName = name + ext;
	TextFileRead();
	
    sourceText = (char *)mSourceString.c_str();
	sourceLength = (GLint)mSourceString.length();
 
	mShader = glCreateShader(type);
    GL_CHECK("glCreateShader() failed");
	glShaderSource(mShader, 1, (const GLchar **)&sourceText, &sourceLength);
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

XGLShader::XGLShader(std::string n) : shaderName(n), programId(-1) {} 
XGLShader::~XGLShader() { }

bool XGLShader::Compile(std::string name) {
    GLint status;

    if (!mVShader.Compile(name, GL_VERTEX_SHADER))
        throwXGLException("Compiling vertex shader failed.");
    if (!mFShader.Compile(name, GL_FRAGMENT_SHADER))
        throwXGLException("Compiling fragment shader failed.");

	programId = glCreateProgram();
    GL_CHECK("glCreateProgram() failed");

	glAttachShader(programId, mVShader.mShader);
    GL_CHECK("glAttachShader(VERTEX) failed");
	glAttachShader(programId, mFShader.mShader);
    GL_CHECK("glAttachShader(FRAGMENT) failed");
	glBindAttribLocation(programId, 0, "in_Position");
    GL_CHECK("glBindAttribLocation(\"in_Position\") failed");
	glBindAttribLocation(programId, 1, "in_TexCoord");
	GL_CHECK("glBindAttribLocation(\"in_TexCoord\") failed");
	glBindAttribLocation(programId, 2, "in_Normal");
    GL_CHECK("glBindAttribLocation(\"in_Normal\") failed");
	glBindAttribLocation(programId, 3, "in_Color");
    GL_CHECK("glBindAttribLocation(\"in_Color\") failed");

	glLinkProgram(programId);
    GL_CHECK("glLinkProgram() failed");
	glValidateProgram(programId);
    GL_CHECK("glValidateProgram() failed");
    
	glGetProgramiv(programId, GL_LINK_STATUS, &status);
    GL_CHECK("glGetProgramiv() failed");
    if (status != GL_TRUE)
        throwXGLException("compile failed in the link stage");
	glGetProgramiv(programId, GL_VALIDATE_STATUS, &status);
    GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_ATTACHED_SHADERS, &status);
    GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &status);
    GL_CHECK("glGetProgramiv() failed");

	if ((modelUniformLocation = glGetUniformLocation(programId, "model")) == GL_INVALID_VALUE)
		throwXGLException("failed to find the model matrix uniform");

	if ((materialLocations.ambientLocation = glGetUniformLocation(programId, "ambient")) == GL_INVALID_VALUE)
		throwXGLException("failed to find the material ambient uniform");

	if ((materialLocations.diffuseLocation = glGetUniformLocation(programId, "diffuse")) == GL_INVALID_VALUE)
		throwXGLException("failed to find the material diffuse uniform");

	if ((materialLocations.specularLocation = glGetUniformLocation(programId, "specular")) == GL_INVALID_VALUE)
		throwXGLException("failed to find the material specular uniform");

	if ((materialLocations.shininessLocation = glGetUniformLocation(programId, "shininess")) == GL_INVALID_VALUE)
		throwXGLException("failed to find the material shininess uniform");

	GLuint idx = glGetUniformBlockIndex(programId, "MatrixData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx == GL_INVALID_INDEX) {
		xprintf("MatrixData not found in '%s'\n", name.c_str());
	}
	else {
		// assign to uniform block 0
		glUniformBlockBinding(programId, idx, 0);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	idx = glGetUniformBlockIndex(programId, "LightData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx != GL_INVALID_INDEX) {
		// assign to uniform block 1
		glUniformBlockBinding(programId, idx, 1);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	idx = glGetUniformBlockIndex(programId, "MaterialData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx != GL_INVALID_INDEX) {
		// assign to uniform block 1
		glUniformBlockBinding(programId, idx, 2);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	glUseProgram(programId);
	GL_CHECK("glUseProgram() failed");

	//DebugPrintf("XGLShader::Compile(%s) shader #: %d\n", name.c_str(), shader);
    return true;
}

bool XGLShader::CompileCompute(std::string name) {
	GLint status;

	xprintf("XGLShader::CompileCompute(%s)\n", name.c_str());

	if (!mCShader.Compile(name, GL_COMPUTE_SHADER))
		throwXGLException("Compiling compute shader failed.");

	programId = glCreateProgram();
	GL_CHECK("glCreateProgram() failed");

	glAttachShader(programId, mCShader.mShader);
	GL_CHECK("glAttachShader(COMPUTE) failed");

	glLinkProgram(programId);
	GL_CHECK("glLinkProgram() failed");
	glValidateProgram(programId);
	GL_CHECK("glValidateProgram() failed");

	glGetProgramiv(programId, GL_LINK_STATUS, &status);
	GL_CHECK("glGetProgramiv() failed");
	if (status != GL_TRUE)
		throwXGLException("compile failed in the link stage");
	glGetProgramiv(programId, GL_VALIDATE_STATUS, &status);
	GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_ATTACHED_SHADERS, &status);
	GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &status);
	GL_CHECK("glGetProgramiv() failed");

	glUseProgram(programId);
	GL_CHECK("glUseProgram() failed");

	glUniform1i(glGetUniformLocation(programId, "destTex"), 0);
	GL_CHECK("glUniform1i() failed");

	return true;
}

void XGLShader::InfoLog() {
	int infoLogLength =0;
	int nWritten;
	char infoLog[8192];

	glValidateProgram(programId);
    GL_CHECK("glValidateProgram() failed");
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &infoLogLength);
    GL_CHECK("glGetProgramiv() failed");
	if (infoLogLength <= sizeof(infoLog)) {
		glGetProgramInfoLog(programId, infoLogLength, &nWritten, infoLog);
		xprintf("%s\n\n", infoLog);
	}
	else
		xprintf("There's an info log, but it's too big!\n");
}

void XGLShader::Use() const {
    GLint currentProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&currentProgram);
    GL_CHECK("glGetIntegerv() failed");
	if (programId != currentProgram) {
		glUseProgram(programId);
        GL_CHECK("glUseProgram(programId) failed");
    }
}

void XGLShader::UnUse() const {
	glUseProgram(0);
    GL_CHECK("glUseProgram(0) failed");
}

void XGLShader::Reshape(int w, int h){
	xprintf("XGLShader::Reshape(%d,%d) for '%s'\n", w, h, shaderName.c_str());
}

std::string XGLShader::Name() const {
	return shaderName;
}

GLint XGLShader::Attrib(std::string name) {
	GLint attrib = glGetAttribLocation(programId, name.c_str());
	if (attrib == -1)
		throwXGLException("Vertex attribute '"+name+"' not found in '" + shaderName + "'");

	return attrib;
}

GLint XGLShader::Uniform(std::string name) {
	GLint uniform = glGetUniformLocation(programId, name.c_str());
	if (uniform == -1)
		throwXGLException("Program uniform '" + name + "' not found in '" + shaderName + "'");
	return uniform;
}

void XGLShader::SetUniform(std::string name, GLint v){
	glUniform1i(glGetUniformLocation(programId, name.c_str()), v);
	GL_CHECK("glUniform1i() failed");
}
