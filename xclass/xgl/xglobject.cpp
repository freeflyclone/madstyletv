#include "xglobject.h"

namespace {
	std::map<std::string, int> namesMap;
}

XGLObject::XGLObject(std::string n) : parent(NULL) {
	uchildren = std::make_unique<XGLObjectChildren>();
}

XGLObject::~XGLObject() {
}

void XGLObject::SetName(std::string n) {
	int i = namesMap[n]++;
	name = n + std::to_string(i);
}

void XGLObject::AddChild(XGLObject *c) {
	c->parent = this;
	uchildren->push_back(c);
}

void XGLObject::DumpChildren()
{
	XGLObjectChildren::iterator ci;
	int i = 0;

	for (ci = uchildren->begin(); ci != uchildren->end(); ci++, i++) {
		XGLObject *xo = *ci;
		xprintf("Child: %d, name: %s\n", i, xo->name.c_str());
	}
}
