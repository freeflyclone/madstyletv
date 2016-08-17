/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** definitions of XConfig objects:
**
****************************************************************************/
#ifndef XCONFIG_H
#define XCONFIG_H

#include <list>
#include <expat.h>

typedef std::pair<std::string, std::string> XConfigKeyValuePair;

class XConfig {
public:
    XConfig();
    ~XConfig();

private:
    void Start(const char *name, const char *args[]);
    void Value(const char *val, int len);
    void End(const char *name);

    static void start(void *userData, const char *name, const char *args[]);
    static void value(void *userData, const char *val, int len);
    static void end(void *userData, const char *name);

    XML_Parser parser;

    std::vector<XConfigKeyValuePair> nodes;
    std::vector<std::string> nodeStack;

    int level;
    int numLeaves;
};

#endif