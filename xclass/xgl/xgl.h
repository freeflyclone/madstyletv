/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved. (for now)
**
** Description:
**  XGL aims to be a minimal C++11, cross-platform, and modern OpenGL
**  framework.
**
**  The goal is to eventually gift to the open source community in the 
**  form of a tutorial web-site with example code.
**
**  This framework was developed in-between gainful employment, and is the
**  3rd or 4th rewrite from scrath of ideas that have been coursing
**  through the grey-matter for well over a decade.
**
**  It would simply not have been possible without contributions from 
**  enumerable sources: web sites, forums, co-workers, family and friends.
****************************************************************************/
#ifndef XGL_H
#define XGL_H

#ifdef _WIN32
	#include "glew.h"
	#include "wglew.h"
	#include <windows.h>
	// If we're compiling with Visual Studio 2015 (or greater)
	// add in the "legacy_stdio_definitions.lib" to avoid 
	// unresolved stdio externals.
	// I hate that this is here, but I didn't find a way to 
	// have the same check in the linker command stuff within
	// the IDE.
	#if _MSC_VER >= 1900
		#pragma comment(lib,"legacy_stdio_definitions")
	#endif
#elif _APPLE_
	#include <OpenGL/gl3.h>
	#define __gl_h_
	#include <GLUT/glut.h>
#else
	#include "glxew.h"
	#include "glew.h"
	#include <GL/gl.h>
#endif


#include <string>
#include <memory>
#include <map>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <vector>
#include <cmath>
#include "glm.hpp"
#include "matrix_transform.hpp"
#include "type_ptr.hpp"

#include "SOIL.h"
#include "ft2build.h"
#include FT_FREETYPE_H

#include "xutils.h"
#include "xthread.h"
#include "xglobject.h"
#include "xconfig.h"
#include "xglexcept.h"
#include "xpath.h"
#include "xglprimitives.h"
#include "xglfont.h"
#include "xglshader.h"
#include "xgluniform.h"
#include "xgllight.h"
#include "xglmaterial.h"
#include "xglbuffer.h"
#include "xglshapes.h"
#include "xglcamera.h"
#include "xglprojector.h"

// want to reference XGLShader by it's name, so use std::map for that
typedef std::map<std::string, GLint> XGLTextureMap;
typedef std::map<std::string, XGLShader *> XGLShaderMap;

// define types for std::vector of shapes ptrs, and the std::map of shapes-lists-by-shader
typedef std::vector<XGLShape *> XGLShapeList;
typedef std::map<std::string, XGLShapeList *> XGLShapesMap;

// define a type for passing a lambda that creates an XGLShape as an argument
typedef std::function<XGLShape *()> XGLNewShapeLambda;

// write code to set these before creating XGL instance, for XGLException messages
// and to locate assets (ie: shaders) in the local filesystem
extern std::string currentWorkingDir;
extern std::string pathToAssets;

// Manage a GL context, platform independently
class XGL : public XGLObject
{
public:
#ifdef USE_GLUT
    // this one is for GLUT applications (probably Linux and/or Mac OS)
    XGL(int *argcp, char **argv);
#endif

    // this one is for Windows (wgl)
    XGL(void);

    virtual ~XGL();

	virtual void Display();
	virtual void Idle() {};

    void AddShape(std::string shaderName, XGLNewShapeLambda fn);
    void IterateShapesMap();

	// query the OpenGL context for various implementation limits, and dump output
	void QueryContext();

	// some of these might be dynamic, but that'll be later
	XGLLights lights;

    XGLShader *GetShader() { return currentShader; };
	XGLShaderMatrixData *GetMatrix() { return &shaderMatrix; }

    XConfig config;

    // all the scene objects, mapped by XGLShader name
    XGLShapesMap shapes;

    // encapsulate the camera and projection tranforms (view,perspectiv matrices)
    XGLCamera camera;
    XGLProjector projector;

    // Shader stack is global per context
    XGLShaderMap shaderMap;

    float clock;

    // for Singleton goodness. There can be only one XGL
    static std::shared_ptr<XGL>getInstance();

private:
    // this is returned by GetShader().  Use of GetShader() feels funky, like my structure design blows chunks.
    XGLShader *currentShader;
	XGLShaderMatrixData shaderMatrix;

    // these are used by GLUT implementations
#ifdef USE_GLUT
    static void display();
	static void reshape(int w, int h);
	static void idle();
#endif
};

#endif // XGL_H
