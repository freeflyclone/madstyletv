#include "xglobject.h"

namespace {
	std::map<std::string, int> namesMap;
	XGLObjectPtr foundObject;
}

XGLObject::XGLObject(std::string n) : parent(NULL) {
	uchildren = std::make_unique<XGLObjectChildren>();
}

XGLObject::~XGLObject() {
}

void XGLObject::SetName(std::string n, bool makeUnique) {
	if (makeUnique) {
		int i = namesMap[n]++;
		name = n + std::to_string(i);
	}
	else {
		name = n;
	}
}

void XGLObject::AddChild(XGLObject *c) {
	c->parent = this;
	uchildren->push_back(c);
}

void XGLObject::DumpChildren()
{
	static int level = 0;
	int i = 0;

	for (auto child : Children()) {
		xprintf("Level: %d, Child: %d: '%s'\n", level, i++, child->name.c_str());
		level++;
		child->DumpChildren();
		level--;
	}
}

XGLObjectPtr XGLObject::FindObject(std::string name) {
	static int level = 0;
	
	if (level == 0)
		foundObject = nullptr;
	
	for (auto child : Children()) {
		if (child->name == name) {
			foundObject = child;
			break;
		}
		level++;
		child->FindObject(name);
		level--;
	}

	return foundObject;
}