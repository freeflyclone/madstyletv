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
	mFileName = name + (const char *)((type==GL_VERTEX_SHADER)?".vert":".frag");
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

XGLShader::XGLShader() : programId(-1) {} 
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
        GL_CHECK("compile failed in the link stage");
	glGetProgramiv(programId, GL_VALIDATE_STATUS, &status);
    GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_ATTACHED_SHADERS, &status);
    GL_CHECK("glGetProgramiv() failed");
	glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &status);
    GL_CHECK("glGetProgramiv() failed");

	GLuint idx = glGetUniformBlockIndex(programId, "ShaderMatrixData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx == GL_INVALID_INDEX) {
		xprintf("ShaderMatrixData not found in '%s'\n", name.c_str());
	}
	else {
		// assign to uniform block 0
		glUniformBlockBinding(programId, idx, 0);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	idx = glGetUniformBlockIndex(programId, "LightData");
	GL_CHECK("glGetUniformBlockIndex() failed");
	if (idx == GL_INVALID_INDEX) {
		xprintf("LightData not found in '%s'\n", name.c_str());
	}
	else {
		// assign to uniform block 1
		glUniformBlockBinding(programId, idx, 1);
		GL_CHECK("glUniformBlockBinding() failed");
	}

	glUseProgram(programId);

	XGLLights lights = XGL::getInstance()->lights;
	XGLLight light = lights.back();

	glUniform3fv(glGetUniformLocation(programId, "light.position"), 1, (GLfloat*)glm::value_ptr(light.position));
	GL_STATUS();

	glUniform3fv(glGetUniformLocation(programId, "light.intensities"), 1, (GLfloat*)glm::value_ptr(light.diffuse));
	GL_STATUS();

	glUniform1f(glGetUniformLocation(programId, "light.attenuation"), light.attenuation);
	GL_STATUS();

	glUniform1f(glGetUniformLocation(programId, "light.ambientCoefficient"), light.ambientCoefficient);
	GL_STATUS();

	glUniform3fv(glGetUniformLocation(programId, "materialSpecularColor"), 1, (GLfloat*)glm::value_ptr(white));
	GL_STATUS();

	glUniform1f(glGetUniformLocation(programId, "materialShininess"), 100.0f);
	GL_STATUS();

	//DebugPrintf("XGLShader::Compile(%s) shader #: %d\n", name.c_str(), shader);
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

void XGLShader::Use() {
    GLint currentProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&currentProgram);
    GL_CHECK("glGetIntegerv() failed");
	if (programId != currentProgram) {
		glUseProgram(programId);
        GL_CHECK("glUseProgram(programId) failed");
    }
}

void XGLShader::UnUse() {
	glUseProgram(0);
    GL_CHECK("glUseProgram(0) failed");
}

void XGLShader::Reshape(int w, int h){
	xprintf("XGLShader::Reshape(%d,%d) for '%s'\n", w, h, shaderName.c_str());
}

std::string XGLShader::Name() {
	return shaderName;
}

GLint XGLShader::Id() {
	return programId;
}

GLint XGLShader::Attrib(std::string name) {
	GLint attrib = glGetAttribLocation(programId, name.c_str());
	if (attrib == -1)
		throwXGLException("Vertex attribute not found: " + name);
	return attrib;
}

GLint XGLShader::Uniform(std::string name) {
	GLint uniform = glGetUniformLocation(programId, name.c_str());
	if (uniform == -1)
		throwXGLException("Shader uniform not found: " + name);
	return uniform;
}
