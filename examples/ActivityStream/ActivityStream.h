#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class ASLink : public json
{
public:
	ASLink();
};

