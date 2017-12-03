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
#include <random>
#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <type_traits>
#include "glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "matrix_transform.hpp"
#include "type_ptr.hpp"

#include "SOIL.h"
#include "ft2build.h"
#include FT_FREETYPE_H

#include "xclasses.h"
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
#include "xglcamera.h"
#include "xglprojector.h"
#include "xglframebuffer.h"
#include "xglpixelbuffer.h"
#include "xglshapes.h"
#include "xglgui.h"
#include "xglhmd.h"

// want to reference XGLShader by it's name, so use std::map for that
typedef std::map<std::string, XGLShader *> XGLShaderMap;

// define types for std::vector of shapes ptrs, and the std::map of shapes-lists-by-shader
typedef std::vector<XGLShape *> XGLShapeList;
typedef std::map<std::string, XGLShapeList *> XGLShapesMap;

// layers of XGLShapesMap, for sorting of scene objects to achieve rendering effects
// such as sky box, transparency, and others as yet unconceived.
typedef std::vector<XGLShapesMap*> XGLShapeLayers;

// Global definition of animate function, and vector thereof
typedef std::function<void(float)> AnimationFn;
typedef std::vector<AnimationFn> AnimationFunctions;

// write code to set these before creating XGL instance, for XGLException messages
// and to locate assets (ie: shaders) in the local filesystem
extern std::string currentWorkingDir;
extern std::string pathToAssets;

extern XGLFont font;

// Manage a GL context, platform independently
class XGL : public XObject, public XInput
{
public:
    // this one is for Windows (wgl)
    XGL(void);

    virtual ~XGL();

	void PreRender();
	void PostRender();

	virtual void Animate();
	virtual bool Display();
	virtual void RenderScene(XGLShapesMap *);
	virtual void Idle() {};

	void InitHmd();

	void AddPreRenderFunction(AnimationFn fn) { preRenderFunctions.push_back(fn); }
	void AddPostRenderFunction(AnimationFn fn) { postRenderFunctions.push_back(fn); }

	XGLShape* CreateShape(XGLShapesMap *s, std::string shaderName, XGLNewShapeLambda fn);
	XGLShape* CreateShape(std::string shaderName, XGLNewShapeLambda fn, int layer = defaultLayer);
	void AddShape(std::string shaderName, XGLNewShapeLambda fn, int layer = defaultLayer);
	void AddGuiShape(std::string shaderName, XGLNewShapeLambda fn);
	void IterateShapesMap();

	// query the OpenGL context for various implementation limits, and dump output
	void QueryContext();

	// some of these might be dynamic, but that'll be later
	XGLLights lights;

    XGLShader *GetShader() { return currentShader; };
	XGLShader *GetShader(std::string name) { return shaderMap[pathToAssets + "/" + name]; }
	XGLShaderMatrixData *GetMatrix() { return &shaderMatrix; }

	void RenderGui(bool enable) { renderGui = enable; }
	bool GuiIsActive() { return renderGui; }
	XGLGuiManager *GetGuiManager() { return guiManager; }
	bool GuiResolveMouseEvent(XGLShape *, int, int, int);

	XConfig config;

	// all the scene objects and GUI objects , mapped by XGLShader name
    //XGLShapesMap shapes;
	XGLShapesMap guiShapes;

	// Layers allow for sorting groups of object to achieve RenderScene() effects
	// that require sorting, in particular transparency.
	static const int defaultLayer = 1;
	XGLShapeLayers shapeLayers;

    // encapsulate the camera and projection tranforms (view,perspectiv matrices)
    XGLCamera camera;
    XGLProjector projector;

    // Shader stack is global per context
    XGLShaderMap shaderMap;

	XGLSharedPBO *pb;
	XGLSharedFBO *fb;

	float clock;

	XGLShaderMatrixData shaderMatrix;

	// dimensions of our window (it can change)
	int width, height;
	XGLShape *mouseCaptured;
	XGLShape *keyboardFocused;

	void GetPreferredWindowSize(int *width, int *height);
	bool UseHMD() { return useHmd; }
	int GetPreferredSwapInterval() { return preferredSwapInterval; }
	int preferredWidth, preferredHeight;
	bool useHmd;
	int preferredSwapInterval;
	XGLSled *hmdSled;
	XGLHmd *pHmd;

private:
    // this is returned by GetShader().  Use of GetShader() feels funky, like my structure design blows chunks.
    XGLShader *currentShader;
	GLuint matrixUbo, lightUbo;
	bool renderGui;
	XGLGuiManager *guiManager;
	AnimationFunctions preRenderFunctions;
	AnimationFunctions postRenderFunctions;
};

#endif // XGL_H
