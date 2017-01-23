#include "xgl.h"
#include "xassets.h"

XAssets::XAssets(std::string file) {
	std::wifstream infile(file);

	if (infile.is_open()) {
		std::wstringstream tmp;
		tmp << infile.rdbuf();
		infile.close();

		root = JSON::Parse(tmp.str().c_str());

		if (root != NULL) {
			if (!(root->IsObject()))
				xprintf("XAssets::XAssets() root is not an object!\n");
		}
		else
			xprintf("JSON::Parse() failed\n");
	}
}

JSONValue* XAssets::Find(std::wstring name) const {
	std::wstringstream ss(name);
	std::wstring token;
	std::vector<std::wstring>tokens;
	std::vector<std::wstring>::iterator iterator;
	JSONObject object;

	// just in JSONParse() failed
	if (root == NULL)
		return NULL;

	// maybe I should keep root as JSONObject (?)
	object = root->AsObject();

	// split "name" into tokens with '.' as separator
	while (std::getline(ss, token, L'.'))
		tokens.push_back(token);

	for (iterator = tokens.begin(); iterator != tokens.end(); iterator++) {
		if (object.find(iterator->c_str()) != object.end()) {
			if ((iterator + 1) == tokens.end())
				return object[iterator->c_str()];
			else if (object[iterator->c_str()]->IsObject())
				object = object[iterator->c_str()]->AsObject();
		}
	}
	return NULL;
}

std::string XAssets::WideToBytes(const std::wstring &wstr){
	throw std::runtime_error("XAssets::WideToBytes is not implemented!");

	// found this on Stack Overflow, but it doesn't work on gcc 4.9.1
	//return std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t>().to_bytes(wstr);

	// Also found this on Stack Overflow.  gcc 4.9.1 compatibility is unknown.
	//return std::wstring_convert<std::codecvt<wchar_t, char,std::mbstate_t>, wchar_t>().to_bytes(wstr);
}

void XAssets::DebugDump() {
	DebugDump(root);
}

void XAssets::DebugDump(JSONValue *value) {
	static int level = 0;
	std::vector<std::wstring>keys = value->ObjectKeys();
	std::vector<std::wstring>::iterator iterator = keys.begin();

	level++;

	while (iterator != keys.end()){
		JSONValue *keyValue = value->Child((*iterator).c_str());
		if (keyValue){
			xprintf("XAssets::Dump(%d): %S : %S\n", level, (*iterator).c_str(), keyValue->Stringify().c_str());
			if (keyValue->IsObject())
				DebugDump(keyValue);
		}
		iterator++;
	}
	level--;
}

XAssets::~XAssets() {}

