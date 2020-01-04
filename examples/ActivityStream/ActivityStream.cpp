#include "ActivityStream.h"
/*
ASObject::ASObject()
{
	j["@context"] = "https://www.we.org/ns/activitystreams";
	j["type"] = "Object";
	j.emplace("attachment");
	j.emplace("attributedTo");
	j.emplace("audience");
	j.emplace("content");
	j.emplace("context");
	j.emplace("name");
	j.emplace("endTime");
	j.emplace("generator");
	j.emplace("icon");
	j.emplace("image");
	j.emplace("inReply");
	j.emplace("location");
	j.emplace("preview");
	j.emplace("published");
	j.emplace("replies");
	j.emplace("startTime");
	j.emplace("summary");
	j.emplace("tag");
	j.emplace("updated");
	j.emplace("url");
	j.emplace("to");
	j.emplace("bto");
	j.emplace("cc");
	j.emplace("bcc");
	j.emplace("mediaType");
	j.emplace("duration");
};

*/

ASLink::ASLink()
{
	emplace("@context", std::vector<std::string>{"https://www.we.org/ns/activitystreams"});
	emplace("type", "Link");
}

