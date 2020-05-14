/**************************************************************
** HDF5Dev: demonstrate use of HDF5 big data library
**************************************************************/
#include "ExampleXGL.h"

void ExampleXGL::BuildScene() {
	XGLShape* shape;

	AddShape("shaders/000-simple", [&]() { shape = new XGLTriangle(); return shape; });
}