#include "xclasses.h"

namespace {
	std::map<std::string, int> namesMap;
	XObjectPtr foundObject;
}

XObject::XObject(std::string n) {
	uchildren = std::make_unique<XObjectChildren>();
}

XObject::~XObject() {
	xprintf("XObject::~XObject() - dtor: %s\n", name.c_str());
}

void XObject::SetName(std::string n, bool makeUnique) {
	if (makeUnique) {
		int i = namesMap[n]++;
		name = n + std::to_string(i);
	}
	else {
		name = n;
	}
}

void XObject::AddChild(XObjectPtr c) {
	uchildren->push_back(c);
}

void XObject::DumpChildren()
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

XObjectPtr XObject::FindObject(std::string name) {
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