/****************************************************************************
**
** Copyright (C) 2015 Evan Mortimore
** All rights reserved.
**
** definitions of 3D graphics primitive object classes:
**  vertex, normal, color, texture coordinates
**  In other words, that which is manipulated by the XGLBuffer object
**  for rendering with OpenGL retained mode
**
**  defined in this separate file in order to avoid circular references
**  in other XGLxxx header files.
**
** After studying the OpenGL document on vertex shenanigans:
**  https://www.opengl.org/wiki/Vertex_Specification_Best_Practices
** I find that I don't have the patience (today) to fully understand
** what is best in a generic sense.  So... I'm going to implement my best
** guess as I see it today, with an eye to maybe needing to change it in the
** future.  For now, I'm going specify vertex,normal,color and texcoords as
** separate objects, and one object that combines all of them.  Clients
** can therefore blend their own if desired.
****************************************************************************/
#ifndef XGLPRIMITIVES_H
#define XGLPRIMITIVES_H

typedef glm::vec3 XGLVertex;
typedef glm::vec3 XGLNormal;
typedef glm::vec4 XGLColor;
typedef glm::vec2 XGLTexCoord;

struct XGLVertexAttributes {
    XGLVertex v;
    XGLNormal n;
    XGLTexCoord t;
    XGLColor c;
};

struct XGLMaterials
{
    GLfloat ambientColor[4];
    GLfloat diffuseColor[4];
    GLfloat specularColor[4];
    GLfloat specularExponent;
};

struct XGLLightLocations
{
    GLint directionLocation;
    GLint ambientColorLocation;
    GLint diffuseColorLocation;
    GLint specularColorLocation;
};

struct XGLMaterialLocations
{
    GLint ambientColorLocation;
    GLint diffuseColorLocation;
    GLint specularColorLocation;
    GLint specularExponentLocation;
};

typedef GLushort XGLIndex;
#define XGLIndexType GL_UNSIGNED_SHORT

typedef std::vector<XGLVertexAttributes> XGLVertexList;
typedef std::vector<XGLIndex> XGLIndexList;

// GLSL uniform buffer object for transformation matrices.
// Bound to binding slot 0 for all shaders.
typedef struct XGLShaderMatrixData_t {
	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 model;
	glm::mat4 mvp;
} XGLShaderMatrixData;

#endif