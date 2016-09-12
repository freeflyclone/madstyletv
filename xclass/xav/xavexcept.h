/****************************************************************************
**
** Copyright (C) 2010 Evan Mortimore
** All rights reserved.
**
****************************************************************************/
#ifndef XAVEXCEPT_H
#define XAVEXCEPT_H

#include <stdexcept>

class XAVException: public std::runtime_error
{
	std::string msg;

public:
	XAVException(const char *file, const int line, const std::string &arg="") : 
		runtime_error(arg)
	{
		std::ostringstream o;
		o << file << ":" << line << ": " << arg;
		msg = o.str();
	}

	const char *what() const throw () {
		return msg.c_str();
	}
};

#define throwXAVException(x) throw XAVException(__FILE__, __LINE__, x)

#endif // XAVEXCEPT_H
