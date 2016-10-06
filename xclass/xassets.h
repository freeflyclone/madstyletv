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

#include <list>
#include <JSON.h>

class XAssets : public JSONValue {
public:
	XAssets(std::string fileName);
	~XAssets();

	const std::wstring AsString(std::wstring) const;
	bool AsBool(std::wstring) const;
	double AsNumber(std::wstring) const;
	const JSONArray &AsArray(std::wstring) const;
	const JSONObject &AsObject(std::wstring) const;

	void Dump();
private:
	JSONValue *root;
};

#endif