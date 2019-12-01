/**************************************************************
** AssimpDevBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

void ExampleXGL::BuildScene() {
	XGLShape *shape;
	Assimp::Importer importer;

	AddShape
	(
		"shaders/000-simple", 
		[&]() 
		{ 
			shape = new XGLTriangle(); return shape; 
		}
	);
}
