#include "smartlist.h"

namespace {
	std::map<std::string, int> namesMap;
	SmartObjectPtr foundObject;
}


SmartObject::SmartObject(std::string n) {
	SetName(n);
}

SmartObject::~SmartObject() {
	xprintf("SmartObject::~SmartObject() - dtor: %s\n", name.c_str());
}

void SmartObject::SetName(std::string n, bool makeUnique) {
	if (makeUnique) {
		int i = namesMap[n]++;
		name = n + std::to_string(i);
	}
	else {
		name = n;
	}
}

void SmartObject::AddChild(SmartObjectPtr c) {
	children.push_back(c);
}

void SmartObject::DumpChildren()
{
	static int level = 0;
	int i = 0;

	for (auto child : children) {
		xprintf("Level: %d, Child: %d: '%s'\n", level, i++, child->name.c_str());
		level++;
		child->DumpChildren();
		level--;
	}
}

SmartObjectPtr SmartObject::FindObject(std::string name) {
	static int level = 0;

	if (level == 0)
		foundObject = nullptr;

	for (auto child : children) {
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