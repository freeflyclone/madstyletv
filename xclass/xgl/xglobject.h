/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definition of base object class:
**  For a hierarchy of world objects, allow any object to be attached
**  as a child to any other object.
****************************************************************************/
/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definition of base object class from which all others are (could be) derived:
**	Adding this late in the game, which just shows my lack of planning.
**	Intent is to add the ability to name objects in the world with, and
**  provide a hierarchy of objects that is unrestrictive with regard to
**	who can be a parent and who can be a child.
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

class XGLObject;

typedef XGLObject* XGLObjectPtr;
//typedef std::shared_ptr<XGLObject> XGLObjectChild;
typedef std::vector<XGLObjectPtr> XGLObjectChildren;

class XGLObject {
public:
	XGLObject(std::string n = "XGLObject");
	~XGLObject();

	void SetName(std::string n);
	void AddChild(XGLObject *o);

	XGLObjectChildren Children() { return *uchildren; }
	void DumpChildren();

	std::unique_ptr<XGLObjectChildren> uchildren;
	std::string name;
	XGLObjectPtr parent;
};

