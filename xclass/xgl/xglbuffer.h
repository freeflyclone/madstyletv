/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of OpenGL retained mode API objects:
**  vertex attribute arrays, vertex buffer objects, shaders
**  lights, cameras, materials
**  In other words, that which is manipulated directly by OpenGL API calls
**
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
****************************************************************************/
#ifndef XGLBUFFER_H
#define XGLBUFFER_H


class XGLBuffer {
public:
    XGLBuffer();
    virtual void Bind();
    virtual void Unbind();
    virtual void Load(XGLShader *shader, std::vector<XGLVertexAttributes> va, std::vector<XGLIndex> ib = {});
    virtual void AddTexture(std::string name);
	virtual void AddTexture(std::string, int, int, int, GLubyte *, bool flipColors = false);
	virtual void AddTexture(int, int, int);

	virtual XGLVertexAttributes *MapVertexBuffer();
	virtual void UnmapVertexBuffer();

    GLuint vao;
    GLuint vbo;
    GLuint ibo;
	GLint numTextures;
    std::vector<GLuint> texIds;
    GLfloat clock;
	XGLShader *shader;
};

#endif
