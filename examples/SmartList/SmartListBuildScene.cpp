/**************************************************************
** SmartListBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include "ExampleXGL.h"
#include "smartlist.h"

class Foo : public SmartObject {
public:
	Foo() {
		xprintf("Foo::Foo()\n");
	}

	~Foo() {
		xprintf("Foo::~Foo()\n");
	}
};

class Bar : public SmartObject {
public:
	Bar() {
		xprintf("Bar::Bar()\n");
	}

	~Bar() {
		xprintf("Bar::~Bar()\n");
	}
};



void ExampleXGL::BuildScene() {
	XGLShape *shape;

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });

	Foo foo;
	foo.AddChild(std::make_shared<Bar>());
	foo.DumpChildren();
}
