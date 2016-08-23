/****************************************************************************
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** Implementation of OpenGL retained mode objects defined in the
**  corresponding header file.
****************************************************************************/
#include "xgl.h"

XGLBuffer::XGLBuffer() :
    ibo(0),
    vao(0),
    vbo(0),
    texId(0)
{
    glGenVertexArrays(1, &vao);
    GL_CHECK("glGenVertexArrays() failed");
    glGenBuffers(1, &vbo);
    GL_CHECK("glGenBuffers() failed");
    glGenBuffers(1, &ibo);
    GL_CHECK("glGenBuffers() failed");

    // use the Singleton reference here to get the current shader,
    // and then immediately bind it so setting of uniforms
    // will work
	std::shared_ptr<XGL> xgl = XGL::getInstance();
	program = xgl->GetShader()->shader;
	Bind();
}

void XGLBuffer::Bind(){
    glBindVertexArray(vao);
    GL_CHECK("glBindVertexArray() failed");
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    GL_CHECK("glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo) failed");
    glUseProgram(program);
    GL_CHECK("glUseProgram() failed");

    if (texId != 0){
        // setup the texture unit to 0
        glProgramUniform1i(program, glGetUniformLocation(program, "texUnit"), 0);
        GL_CHECK("glProgramUniform1i() failed");
        glActiveTexture(GL_TEXTURE0);
        GL_CHECK("glActiveTexture() failed");
        glBindTexture(GL_TEXTURE_2D, texId);
        GL_CHECK("glBindTexture() failed");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        GL_CHECK("glTexParameteri() failed");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GL_CHECK("glTexParameteri() failed");
    }
}

void XGLBuffer::Unbind(){
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    GL_CHECK("glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo) failed");
    glBindVertexArray(0);
    GL_CHECK("glBindVertexArray() failed");
    glUseProgram(0);
    GL_CHECK("glUseProgram() failed");
}

XGLVertexAttributes *XGLBuffer::MapVertexBuffer() {
	// bind "vbo" to ensure we map the correct object buffer with glMapBuffer()
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	GL_CHECK("glBindBuffer() failed");

	XGLVertexAttributes *vb = (XGLVertexAttributes *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	GL_CHECK("glMapBuffer() failed");
	return vb;
}

void XGLBuffer::UnmapVertexBuffer() {
	// bind "vbo" to ensure we unmap the correct object buffer with glUnmapBuffer()
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	GL_CHECK("glBindBuffer() failed");

	glUnmapBuffer(GL_ARRAY_BUFFER);
	GL_CHECK("glUnmapBuffer() failed");
}

void XGLBuffer::Load(std::vector<XGLVertexAttributes> va, std::vector<XGLIndex> ib){
    Bind();

    // bind "vao"  and "vbo".  This means setting OpenGL state to refer to these objects
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GL_CHECK("glBindBuffer() failed");

    // this loads the actual XGLVertexAttributes into the bound "vbo"
    glBufferData(GL_ARRAY_BUFFER, va.size()*sizeof(XGLVertexAttributes), va.data(), GL_STATIC_DRAW);
    GL_CHECK("glBufferData() failed");

    // Tell the bound Vertex Attribute Object about how the "vbo" is layed out
    // the first arg is an index to XGLVertex. XGLNormal, XGLTexCoord or XGLColor 
    // component of XGLVertexAttributes, in the order they are defined therein.
    //-----------------------------------------
    // All shaders must agree on this layout!!!
    //-----------------------------------------
    // AND!!! The shader compiler must agree as well.
    // (need to manage this better, perhaps GLSL has a mechanism)
    //-----------------------------------------------------------

    // Vertex coords
    glVertexAttribPointer(0, sizeof(XGLVertex) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(XGLVertexAttributes), 0);
    GL_CHECK("glVertexAttribPointer() failed");
	// vertex texCoords
	glVertexAttribPointer(1, sizeof(XGLTexCoord) / sizeof(float), GL_FLOAT, GL_TRUE, sizeof(XGLVertexAttributes), reinterpret_cast<void *>(sizeof(XGLVertex)));
	GL_CHECK("glVertexAttribPointer() failed");
	// vertex normals
	glVertexAttribPointer(2, sizeof(XGLNormal) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(XGLVertexAttributes), reinterpret_cast<void *>(sizeof(XGLVertex) + sizeof(XGLTexCoord)));
    GL_CHECK("glVertexAttribPointer() failed");
    // vertex colors
	glVertexAttribPointer(3, sizeof(XGLColor) / sizeof(float), GL_FLOAT, GL_TRUE, sizeof(XGLVertexAttributes), reinterpret_cast<void *>(sizeof(XGLVertex) + sizeof(XGLNormal) + sizeof(XGLTexCoord)));
    GL_CHECK("glVertexAttribPointer() failed");

    glEnableVertexAttribArray(0);
    GL_CHECK("glEnableVertexAttribArray() failed");
    glEnableVertexAttribArray(1);
    GL_CHECK("glEnableVertexAttribArray() failed");
    glEnableVertexAttribArray(2);
    GL_CHECK("glEnableVertexAttribArray() failed");
    glEnableVertexAttribArray(3);
    GL_CHECK("glEnableVertexAttribArray() failed");

    // "ibo" is always created, but we don't have to use it.
    // However, if the client (derived from XGLShape) passes vector of XGLIndex whose
    // size is greater than zero, clearly the intent is to do indexed drawing.
    // Apple's OpenGL 3.2 is core profile only, index arrays must ALSO be 
    // buffer objects, (IndexBufferObject aka IBO), so set one up for this object
    if (ib.size() > 0){
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        GL_CHECK("glBindBuffer() failed");
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizei)(ib.size()*sizeof(XGLIndex)), (GLvoid *)(ib.data()), GL_STATIC_DRAW);
        GL_CHECK("glBufferData() failed");
    }
}

void XGLBuffer::AddTexture(std::string texName){
    int width, height, channels;
    unsigned char *img;

    if ((img = SOIL_load_image(texName.c_str(), &width, &height, &channels, 0)) == NULL)
        throwXGLException("SOIL_load_image() failed: " + texName);

    AddTexture(texName, width, height, channels, img);

    SOIL_free_image_data(img);
}

void XGLBuffer::AddTexture(std::string texName, int width, int height, int channels, GLubyte *img){
    glGenTextures(1, &texId);
    GL_CHECK("glGenTextures() failed");
    glBindTexture(GL_TEXTURE_2D, texId);
    GL_CHECK("glBindTexture() failed");

    glActiveTexture(GL_TEXTURE0);
    GL_CHECK("glActiveTexture(GL_TEXTURE0) failed");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GL_CHECK("glPixelStorei() failes");

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    GL_CHECK("glTexParameteri() failed");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GL_CHECK("glTexParameteri() failed");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    GL_CHECK("glTexParameteri() failed");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GL_CHECK("glTexParameteri() failed");

    switch (channels){
        case 1:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, img);
            break;
        case 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, width, height, 0, GL_RG, GL_UNSIGNED_BYTE, img);
            break;
        case 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
            break;

        default:
            throwXGLException("Unknown pixel format in XGL::AddTexture");
    }
    GL_CHECK("glTexImage2D() failed");

    XGL::getInstance()->textures.emplace(texName, texId);
    texIds.push_back(texId);
}
