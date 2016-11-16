/****************************************************************************
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
**   https://github.com/MJPA/SimpleJSON
**
** This class is little more than a file-based wrapper around SimpleJSON.
**
** It is intended to be used for things like configuration parameters,
** scene descriptions, edit decision lists and so on.
**
** Note: wstring is used throughout SimpleJSON, thus Unicode is supported.
**       That may be unnecessary, but better safe than sorry.
****************************************************************************/
#ifndef XASSETS_H
#define XASSETS_H

#include "JSON.h"
#include <string>
#include <locale>
#include <iostream>
#include <sstream>

class XAssets : public JSONValue {
public:
	XAssets(std::string fileName);
	~XAssets();

	// find something with a "object.object.[object.]*" style search string.
	JSONValue *Find(std::wstring) const;

	void DebugDump();
	void DebugDump(JSONValue *value);

	std::string WideToBytes(const std::wstring &wstr);

private:
	JSONValue *root;
};

#endif