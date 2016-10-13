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


//---------------------------------------------------
// NOTE: coding the constructor this way causes a seg 
// fault on linux if XCLASS_DIR is not set. It needs 
// to be changed!!!
//---------------------------------------------------
class XConfig : public XAssets {
public:
	//XConfig(std::string fileName = std::getenv("XCLASS_DIR") + std::string("/../assets/config.json")) : XAssets(fileName) {};
	XConfig(std::string fileName = std::getenv("XCLASS_DIR")?std::getenv("XCLASS_DIR"):"." + std::string("/../assets/config.json")) : XAssets(fileName) {};
};

#endif
