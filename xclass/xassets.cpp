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
const std::wstring XAssets::AsString(std::wstring name) const {
	return root->Child(name.c_str())->AsString();
};
bool XAssets::AsBool(std::wstring name) const {
	return root->Child(name.c_str())->AsBool();
};
double XAssets::AsNumber(std::wstring name) const {
	return root->Child(name.c_str())->AsNumber();
};
const JSONArray &XAssets::AsArray(std::wstring name) const {
	return root->Child(name.c_str())->AsArray();
};
const JSONObject &XAssets::AsObject(std::wstring name) const {
	return root->Child(name.c_str())->AsObject();
};

void XAssets::Dump() {
	std::vector<std::wstring>keys = root->ObjectKeys();

	for (size_t i = 0; i < root->CountChildren(); i++) {
		JSONValue *child = root->Child(keys[i].c_str());
		xprintf("%S : %S\n", keys[i].c_str(), child->Stringify(true).c_str());
	}
}

XAssets::~XAssets() {}

