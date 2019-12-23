/**************************************************************
** ActivityStreamBuildScene.cpp
**
** Just to demonstrate instantiation of a "ground"
** plane and a single triangle, with default camera manipulation
** via keyboard and mouse.
**************************************************************/
#include <ostream>

#include "ExampleXGL.h"
#include "ActivityStream.h"
#include "DebugOutput.h"

class ASLink : public json
{
public:
	ASLink();
};

ASLink::ASLink()
{
	emplace("@context", std::vector<std::string>{"https://www.we.org/ns/activitystreams"});
	
	emplace("type","Link");
}


void ExampleXGL::BuildScene() {
	XGLShape *shape;
	ASLink asLink;

	InitStdLog();

	{
		std::ifstream ifs(pathToAssets + "/assets/timeline-public.json");
		ASLink j;
		ifs >> j;
		std::clog << std::setw(2) << j << std::endl;
	}

	asLink["@context"].push_back("https://e-man.tv/.well_known");
	asLink["preview"] = "https://hq.e-man.tv/hls-test.html";

	std::clog << std::setw(2) << asLink << std::endl;

	/*
	if (asLink["preview"].size())
		xprintf("preview: %s\n", asLink["preview"].get<std::string>().c_str());

	for (auto el : asLink["@context"])
		xprintf("@context: %s\n", el.get<std::string>().c_str());
	*/

	AddShape("shaders/000-simple", [&](){ shape = new XGLTriangle(); return shape; });
}
