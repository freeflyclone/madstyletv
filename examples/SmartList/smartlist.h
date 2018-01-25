#ifndef SMARTLIST_H
#define SMARTLIST_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "xutils.h"

class SmartObject;

typedef std::shared_ptr<SmartObject> SmartObjectPtr;
typedef std::vector<SmartObjectPtr> SmartObjectChildren;

class SmartObject {
public:
	SmartObject(std::string n = "SmartObject");
	virtual ~SmartObject();

	void SetName(std::string n, bool makeUnique = true);
	void AddChild(SmartObjectPtr);

	SmartObjectChildren Children() { return children; }
	SmartObjectPtr FindObject(std::string n);

	std::string Name() { return name; }

	void DumpChildren();

private:
	SmartObjectChildren children;
	std::string name;
};



#endif