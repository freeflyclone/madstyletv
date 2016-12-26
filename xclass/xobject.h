/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definition of base object class from which all others are (could be) derived:
**	Adding this late in the game, which just shows my lack of planning.
**	Intent is to add the ability to name objects, and provide a hierarchy 
**  of objects that is unrestrictive with regard to who can be a parent 
**  and who can be a child.
**
**	C++11 smart pointers will be utilized to make the hierarchy behave
**	correctly for world objects of finite life-time.
****************************************************************************/
#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "xutils.h"

class XObject;

typedef XObject* XObjectPtr;
typedef std::vector<XObjectPtr> XObjectChildren;

class XObject {
public:
	XObject(std::string n = "XObject");
	~XObject();

	void SetName(std::string n, bool makeUnique = true);
	void AddChild(XObject *o);

	XObjectChildren Children() { return *uchildren; }
	void DumpChildren();
	XObjectPtr FindObject(std::string n);

	XObjectPtr Parent() { return parent; }
	std::string Name() { return name; }

private:
	std::unique_ptr<XObjectChildren> uchildren;
	std::string name;
	XObjectPtr parent;
};

