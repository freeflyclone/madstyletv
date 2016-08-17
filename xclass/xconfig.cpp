#include "xgl.h"
#include "xconfig.h"

XConfig::XConfig() : parser(NULL), level(0), numLeaves(0) {
	xprintf("XConfig::XConfig()\n");

    parser = XML_ParserCreate("UTF-8");
    if (parser == NULL)
        throw std::runtime_error("XML_ParserCreate() failed");

    XML_SetUserData(parser, this);
    XML_SetElementHandler(parser, start, end);
    XML_SetCharacterDataHandler(parser, value);

    std::ifstream infile("config.xml");
    std::string line;
    while (std::getline(infile, line))
        XML_Parse(parser, line.c_str(), (int)(line.size()), line.size() == 0);

    std::vector<XConfigKeyValuePair>::iterator it;
	xprintf("Found %d nodes\n", nodes.size());
    for (it = nodes.begin(); it != nodes.end(); it++)
		xprintf("   Key: %s, Value: %s\n", it->first.c_str(), it->second.c_str());
}

XConfig::~XConfig() {
    XML_ParserFree(parser);
}

void XConfig::Start(const char *name, const char *args[]){
    if (level)
        nodeStack.push_back(name);
    level++;
}

void XConfig::Value(const char *val, int len){
    std::string value(val, len);
    std::size_t pos;

    // ignore strings that are only whitespace.  
    if ((pos = value.find_first_not_of(" \t\n\f\v\r")) == value.npos)
        return;
    
    // and strip leading whitespace if it exists.
    std::string strippedValue = value.substr(pos);

    std::vector<std::string>::iterator it = nodeStack.begin();
    std::string nodeName;

    for (it = nodeStack.begin(); it != nodeStack.end(); it++){
        nodeName += it->c_str();
        if (*it != nodeStack.back())
            nodeName += ".";
    }
    XConfigKeyValuePair leaf(nodeName, strippedValue);
    nodes.push_back(leaf);
    numLeaves++;
}

void XConfig::End(const char *name){
    std::string nodeName(name);

    // we don't push the name on the first level, because it's the root of the 
    // tree, and therefore always the same, and thus essentially useless.
    if (level == 1)
        return;
    
    // if this is a leaf node do a "end of leaf node" thing
    if (numLeaves)
        numLeaves--;

    level--;
    nodeStack.pop_back();
}

void XConfig::start(void *userData, const char *name, const char *args[]){
    static_cast<XConfig *>(userData)->Start(name, args);
}

void XConfig::value(void *userData, const char *val, int len){
    static_cast<XConfig *>(userData)->Value(val,len);
}

void XConfig::end(void *userData, const char *name){
    static_cast<XConfig *>(userData)->End(name);
}
