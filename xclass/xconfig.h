/****************************************************************************
**
** Copyright (C) 2016 Evan Mortimore
** All rights reserved.
**
** XConfig used to be XML based, but it's been switched to XAssets.
** 
** See xassets.h for further details.
****************************************************************************/
#ifndef XCONFIG_H
#define XCONFIG_H

#include <xassets.h>

class XConfig : public XAssets {
public:
	// guard against XCLASS_DIR environment variable not being set, as that causes a seg fault
	XConfig(std::string fileName = std::getenv("XCLASS_DIR")?std::getenv("XCLASS_DIR"):"." + std::string("/../assets/config.json")) : XAssets(fileName) {};
};

#endif
